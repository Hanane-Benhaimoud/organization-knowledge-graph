import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from jsonschema import validate


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RUN_ID = "ff7e5205b7ffa21f"

CHUNK_ID = (
    "4f0358fd53b519ce50fe1920ff90d4ed"
    "01a87a6e68e58b7a87f997154d28d2ba"
)

RUN_DIR = ROOT / "data" / "silver" / "llm" / RUN_ID
RESPONSES = RUN_DIR / "responses"
PARENT_RESPONSE = RESPONSES / f"{CHUNK_ID}.json"

HDFS_CHUNKS = (
    "/data/orggraph/silver/preparation/wikipedia_chunks.jsonl"
)

MAX_CHARS = 900
OVERLAP = 100


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def split_passage(text):
    start = 0

    while start < len(text):
        end = min(start + MAX_CHARS, len(text))

        if end < len(text):
            lower = start + MAX_CHARS // 2

            for separator in ("\n", ". ", "; ", " "):
                position = text.rfind(separator, lower, end)

                if position != -1:
                    end = position + len(separator)
                    break

        piece = text[start:end].strip()

        if piece:
            yield piece

        if end == len(text):
            break

        next_start = max(start + 1, end - OVERLAP)

        # Évite de commencer au milieu d'un mot.
        while (
            next_start < end
            and next_start > 0
            and not text[next_start - 1].isspace()
        ):
            next_start += 1

        start = next_start


def read_parent_chunk():
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "namenode",
            "hdfs", "dfs", "-cat", HDFS_CHUNKS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    matches = []

    for line in result.stdout.decode("utf-8").splitlines():
        if line.strip():
            chunk = json.loads(line)

            if chunk["chunk_id"] == CHUNK_ID:
                matches.append(chunk)

    if len(matches) != 1:
        raise RuntimeError(
            f"Un segment unique était attendu, trouvé : {len(matches)}"
        )

    return matches[0]


def main():
    settings = json.loads(
        (RUN_DIR / "settings.json").read_text(encoding="utf-8")
    )

    # Ne recommence pas une réparation déjà terminée.
    if PARENT_RESPONSE.exists():
        existing = json.loads(
            PARENT_RESPONSE.read_text(encoding="utf-8")
        )

        if existing.get("repair", {}).get("method") == "subchunk_extraction":
            choice = existing["response"]["choices"][0]

            if choice["finish_reason"] == "stop":
                validate(
                    json.loads(choice["message"]["content"]),
                    settings["schema"],
                )
                print("Ce segment a déjà été réparé. Aucun appel effectué.")
                return

    parent = read_parent_chunk()
    pieces = list(split_passage(parent["text"]))

    # Cache séparé des réponses normales.
    repair_identity = {
        "parent_chunk_id": CHUNK_ID,
        "settings": settings,
        "pieces": pieces,
        "method": "subchunk_extraction_v1",
    }

    repair_id = hashlib.sha256(
        json.dumps(
            repair_identity,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]

    repair_dir = (
        RESPONSES / "_repairs" / CHUNK_ID / repair_id
    )

    save_json(repair_dir / "manifest.json", repair_identity)

    key = os.getenv("GROQ_API_KEY", "").strip()

    if not key:
        raise ValueError("Configure GROQ_API_KEY dans .env.")

    client = Groq(
        api_key=key,
        timeout=120,
        max_retries=0,
    )

    all_relations = []
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    print(f"Article : {parent['title']}")
    print(f"Section : {parent['section_heading']}")
    print(f"Sous-segments : {len(pieces)}", flush=True)

    for index, piece in enumerate(pieces, 1):
        path = repair_dir / f"part_{index:03d}.json"

        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            print(f"{index}/{len(pieces)} : cache réutilisé", flush=True)

        else:
            print(
                f"{index}/{len(pieces)} : nouvel appel Groq",
                flush=True,
            )

            response = client.chat.completions.create(
                model=settings["model"],
                temperature=settings["temperature"],
                reasoning_effort=settings["reasoning_effort"],
                max_completion_tokens=settings["max_completion_tokens"],
                messages=[
                    {
                        "role": "system",
                        "content": settings["system_prompt"],
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "article_title": parent["title"],
                                "section_heading": parent["section_heading"],
                                "language": parent["language"],
                                "text": piece,
                            },
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

            raw = response.model_dump(mode="json")
            save_json(path, raw)

            # Même pause que dans le traitement habituel.
            import time
            time.sleep(15)

        choice = raw["choices"][0]

        if choice["finish_reason"] != "stop":
            raise RuntimeError(
                f"Sous-segment {index} incomplet : "
                f"{choice['finish_reason']}. "
                "Les autres sous-réponses restent sauvegardées. "
                "Ne relance pas en boucle."
            )

        data = json.loads(choice["message"]["content"])
        validate(data, settings["schema"])

        # Les contrôles habituels seront appliqués par groq_extract.
        # On conserve ici toutes les propositions complètes,
        # y compris celles qui pourront être rejetées ensuite.
        all_relations.extend(data["relations"])

        part_usage = raw.get("usage") or {}

        for field in usage:
            usage[field] += int(part_usage.get(field) or 0)

    # Regroupe uniquement les propositions exactement identiques.
    unique = {}

    for relation in all_relations:
        identity = json.dumps(
            relation,
            ensure_ascii=False,
            sort_keys=True,
        )
        unique[identity] = relation

    assembled = {"relations": list(unique.values())}
    validate(assembled, settings["schema"])

    saved = {
        "chunk_id": CHUNK_ID,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "repair": {
            "method": "subchunk_extraction",
            "repair_id": repair_id,
            "parts": len(pieces),
            "raw_responses_directory": str(
                repair_dir.relative_to(ROOT)
            ),
            "note": (
                "Résultat assemblé à partir de sous-réponses complètes ; "
                "ce fichier n'est pas une réponse API brute unique."
            ),
        },
        "response": {
            "model": settings["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        assembled,
                        ensure_ascii=False,
                    ),
                },
            }],
            "usage": usage,
        },
    }

    # Préserve la réponse précédente avant de publier la réparation.
    if PARENT_RESPONSE.exists():
        archive = RESPONSES / "_incomplete"
        archive.mkdir(exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        PARENT_RESPONSE.rename(
            archive / f"{CHUNK_ID}_{timestamp}.json"
        )

    save_json(PARENT_RESPONSE, saved)

    print("\nRÉPARATION TERMINÉE")
    print(f"Propositions réunies : {len(unique)}")
    print("Les autres segments n'ont pas été modifiés.")
    print("Relance maintenant groq_extract avec --limit 0.")


if __name__ == "__main__":
    main()