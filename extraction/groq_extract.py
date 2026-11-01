import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from jsonschema import validate


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HDFS_INPUT = (
    "/data/orggraph/silver/preparation/wikipedia_chunks.jsonl"
)

# On reprend explicitement la version déjà utilisée.
DEFAULT_RUN_ID = "ff7e5205b7ffa21f"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_chunks():
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "namenode",
            "hdfs", "dfs", "-cat", HDFS_INPUT,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    chunks = [
        json.loads(line)
        for line in result.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants de segments dupliqués.")

    return sorted(
        chunks,
        key=lambda chunk: (
            chunk["section_index"] != 0,
            chunk["document_id"],
            chunk["section_index"],
            chunk["part_index"],
        ),
    )


def check_relations(data, text, schema):
    validate(instance=data, schema=schema)

    accepted = []
    rejected = []

    for relation in data["relations"]:
        reason = None
        kind = relation["relation"]

        if (
            not relation["subject"].strip()
            or not relation["object"].strip()
        ):
            reason = "empty_endpoint"

        elif not relation["evidence"].strip():
            reason = "empty_evidence"

        elif relation["evidence"] not in text:
            reason = "evidence_not_exact_substring"

        elif (
            relation["subject_type"] != "person"
            and kind in {
                "CEO_OF",
                "CHAIRPERSON_OF",
                "BOARD_MEMBER_OF",
                "EMPLOYED_BY",
            }
        ):
            reason = "invalid_subject_type"

        elif (
            relation["object_type"] != "organization"
            and kind != "OWNED_BY"
        ):
            reason = "invalid_object_type"

        elif (
            kind in {"PARENT_OF", "OWNED_BY", "PUBLISHED_BY"}
            and relation["subject_type"] != "organization"
        ):
            reason = "invalid_subject_type"

        if reason:
            rejected.append({
                "reason": reason,
                "relation": relation,
            })
        else:
            accepted.append(relation)

    return accepted, rejected


def error_details(exc, key):
    body = getattr(exc, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}

    if not isinstance(details, dict):
        details = {}

    message = str(
        details.get("message")
        or getattr(exc, "message", None)
        or type(exc).__name__
    )

    return {
        "type": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None),
        "code": details.get("code"),
        "message": message.replace(key, "[REDACTED]")[:2000],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Nombre maximal de nouveaux appels Groq.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=15,
        help="Pause en secondes après un nouvel appel réussi.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Dossier de l'extraction existante à reprendre.",
    )

    args = parser.parse_args()

    if args.limit < 0 or args.delay < 0:
        parser.error("--limit et --delay doivent être positifs ou nuls.")

    # Empêche un chemin arbitraire via --run-id.
    if (
        len(args.run_id) != 16
        or any(c not in "0123456789abcdef" for c in args.run_id)
    ):
        parser.error("--run-id doit être un identifiant hexadécimal de 16 caractères.")

    output = ROOT / "data" / "silver" / "llm" / args.run_id
    responses = output / "responses"
    settings_path = output / "settings.json"

    if not settings_path.exists() or not responses.is_dir():
        raise FileNotFoundError(
            f"Extraction existante introuvable : {output}\n"
            "Aucun nouveau dossier n'a été créé et aucun appel "
            "Groq n'a été effectué."
        )

    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    required = {
        "model",
        "system_prompt",
        "schema",
        "prompt_version",
        "temperature",
        "max_completion_tokens",
        "reasoning_effort",
    }

    if not required.issubset(settings):
        raise ValueError(
            "settings.json est incomplet. Aucun appel effectué."
        )

    # Vérifie que les paramètres correspondent bien au dossier.
    calculated_id = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    if calculated_id != args.run_id:
        raise ValueError(
            "Les paramètres enregistrés ne correspondent pas au run_id. "
            "Aucun appel effectué."
        )

    key = os.getenv("GROQ_API_KEY", "").strip()

    if args.limit > 0 and not key:
        raise ValueError("Configure GROQ_API_KEY dans .env.")

    chunks = read_chunks()

    cached_count = sum(
        (responses / f"{chunk['chunk_id']}.json").exists()
        for chunk in chunks
    )

    print(f"Extraction reprise : {args.run_id}", flush=True)
    print(f"Modèle enregistré : {settings['model']}", flush=True)
    print(
        f"Réponses déjà sauvegardées : {cached_count} "
        "(aucun appel API pour ces réponses)",
        flush=True,
    )
    print(
        f"Maximum de nouveaux appels : {args.limit}\n",
        flush=True,
    )

    # Désactive les nouvelles tentatives automatiques du SDK.
    client = (
        Groq(api_key=key, timeout=120, max_retries=0)
        if args.limit > 0
        else None
    )

    report = {
        "run_id": args.run_id,
        "model": settings["model"],
        "chunks_total": len(chunks),
        "new_calls": 0,
        "new_responses": 0,
        "cached_responses": 0,
        "chunks_validated": 0,
        "accepted_relations": 0,
        "rejected_relations": 0,
        "error": None,
    }

    relations_tmp = output / "relations.jsonl.tmp"
    rejected_tmp = output / "rejected_relations.jsonl.tmp"

    with (
        relations_tmp.open("w", encoding="utf-8") as valid_file,
        rejected_tmp.open("w", encoding="utf-8") as reject_file,
    ):
        for chunk in chunks:
            response_path = responses / f"{chunk['chunk_id']}.json"

            try:
                if response_path.exists():
                    # IMPORTANT : pas d'appel Groq dans cette branche.
                    # Une réponse illisible provoque un arrêt,
                    # jamais un nouvel appel automatique.
                    saved = json.loads(
                        response_path.read_text(encoding="utf-8")
                    )

                    if saved.get("chunk_id") != chunk["chunk_id"]:
                        raise ValueError(
                            "Le cache ne correspond pas au segment."
                        )

                    report["cached_responses"] += 1

                else:
                    if report["new_calls"] >= args.limit:
                        continue

                    payload = {
                        "article_title": chunk["title"],
                        "section_heading": chunk["section_heading"],
                        "language": chunk["language"],
                        "text": chunk["text"],
                    }

                    report["new_calls"] += 1

                    print(
                        f"Appel {report['new_calls']}/{args.limit} | "
                        f"{chunk['title']} | "
                        f"{chunk['section_heading']}",
                        flush=True,
                    )

                    response = client.chat.completions.create(
                        model=settings["model"],
                        temperature=settings["temperature"],
                        reasoning_effort=settings["reasoning_effort"],
                        max_completion_tokens=(
                            settings["max_completion_tokens"]
                        ),
                        messages=[
                            {
                                "role": "system",
                                "content": settings["system_prompt"],
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "organization_relations",
                                "strict": True,
                                "schema": settings["schema"],
                            },
                        },
                    )

                    saved = {
                        "chunk_id": chunk["chunk_id"],
                        "received_at": utc_now(),
                        "response": response.model_dump(mode="json"),
                    }

                    # Sauvegarde immédiate, avant validation.
                    save_json(response_path, saved)
                    report["new_responses"] += 1
                    time.sleep(args.delay)

                choice = saved["response"]["choices"][0]

                if choice["finish_reason"] != "stop":
                    raise ValueError(
                        "Réponse incomplète : "
                        + str(choice["finish_reason"])
                    )

                result = json.loads(choice["message"]["content"])

                accepted, rejected = check_relations(
                    result,
                    chunk["text"],
                    settings["schema"],
                )

                provenance = {
                    name: value
                    for name, value in chunk.items()
                    if name != "text"
                }

                provenance.update({
                    "run_id": args.run_id,
                    "model": settings["model"],
                    "prompt_version": settings["prompt_version"],
                })

                for relation in accepted:
                    valid_file.write(
                        json.dumps(
                            {
                                "provenance": provenance,
                                "relation": relation,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )

                for rejection in rejected:
                    reject_file.write(
                        json.dumps(
                            {
                                "provenance": provenance,
                                **rejection,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )

                report["chunks_validated"] += 1
                report["accepted_relations"] += len(accepted)
                report["rejected_relations"] += len(rejected)

            except Exception as exc:
                report["error"] = {
                    "chunk_id": chunk["chunk_id"],
                    **error_details(exc, key),
                }

                if isinstance(exc, RateLimitError):
                    report["error"]["retry_after"] = (
                        exc.response.headers.get("retry-after")
                    )

                print("\nArrêt avec sauvegarde :", flush=True)
                print(
                    json.dumps(
                        report["error"],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )
                break

    # Si un problème empêche de relire tout l'ancien cache,
    # on conserve les anciens fichiers agrégés.
    if report["cached_responses"] == cached_count:
        relations_tmp.replace(output / "relations.jsonl")
        rejected_tmp.replace(output / "rejected_relations.jsonl")
        report["aggregate_files_updated"] = True
    else:
        relations_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        report["aggregate_files_updated"] = False

    report["chunks_not_validated"] = (
        len(chunks) - report["chunks_validated"]
    )
    report["finished_at"] = utc_now()

    save_json(output / "extraction_report.json", report)

    print("\nRAPPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDossier : {output}")


if __name__ == "__main__":
    main()