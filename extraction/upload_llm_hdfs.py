import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "ff7e5205b7ffa21f"

LOCAL = ROOT / "data" / "silver" / "llm" / RUN_ID
REMOTE = f"/data/orggraph/silver/llm/{RUN_ID}"

SUMMARY_FILES = [
    "settings.json",
    "relations.jsonl",
    "rejected_relations.jsonl",
    "extraction_report.json",
]


def run(*args):
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def hdfs(*args):
    return run(
        "docker", "compose", "exec", "-T", "namenode",
        "hdfs", "dfs", *args,
    )


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def list_files(path):
    """Renvoie les noms et tailles des fichiers d'un dossier HDFS."""
    listing = hdfs("-ls", path).decode("utf-8")

    files = {}

    for line in listing.splitlines():
        if not line.startswith("-"):
            continue

        fields = line.split(maxsplit=7)

        if len(fields) != 8:
            raise RuntimeError(f"Ligne HDFS inattendue : {line}")

        name = fields[7].rsplit("/", 1)[-1]
        files[name] = int(fields[4])

    return files


def count_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def main():
    # Vérifications avant tout transfert.
    for name in SUMMARY_FILES:
        if not (LOCAL / name).is_file():
            raise FileNotFoundError(LOCAL / name)

    responses_dir = LOCAL / "responses"

    if not responses_dir.is_dir():
        raise FileNotFoundError(responses_dir)

    report = json.loads(
        (LOCAL / "extraction_report.json").read_text(encoding="utf-8")
    )

    if report["run_id"] != RUN_ID:
        raise RuntimeError("Le rapport ne correspond pas au run_id.")

    if report.get("aggregate_files_updated") is not True:
        raise RuntimeError(
            "Les fichiers récapitulatifs ne sont pas à jour. "
            "Exécute d'abord : "
            "python -m extraction.groq_extract --limit 0"
        )

    for name, field in (
        ("relations.jsonl", "accepted_relations"),
        ("rejected_relations.jsonl", "rejected_relations"),
    ):
        actual = count_jsonl(LOCAL / name)

        if actual != report[field]:
            raise RuntimeError(
                f"{name} : {actual} lignes au lieu de {report[field]}. "
                "Arrêt pour éviter d'envoyer un résultat incohérent."
            )

    hdfs("-mkdir", "-p", f"{REMOTE}/responses")

    remote_responses = list_files(f"{REMOTE}/responses")
    remote_summary = list_files(REMOTE)

    local_responses = sorted(responses_dir.glob("*.json"))

    new_responses = []

    for path in local_responses:
        if path.name not in remote_responses:
            new_responses.append(path)

        elif path.stat().st_size != remote_responses[path.name]:
            # Une réponse existante est considérée comme immuable.
            raise RuntimeError(
                f"Une réponse existante diffère en taille : {path.name}. "
                "Vérifie-la avant de continuer."
            )

    changed_summary = []

    for name in SUMMARY_FILES:
        local_data = (LOCAL / name).read_bytes()

        if name not in remote_summary:
            changed_summary.append(name)
            continue

        remote_data = hdfs("-cat", f"{REMOTE}/{name}")

        if sha256(local_data) != sha256(remote_data):
            if name == "settings.json":
                raise RuntimeError(
                    "settings.json diffère de la version HDFS. "
                    "Ne mélange pas des paramètres différents "
                    "sous le même run_id."
                )

            changed_summary.append(name)

    print(f"Réponses locales : {len(local_responses)}")
    print(f"Nouvelles réponses à envoyer : {len(new_responses)}")
    print(f"Fichiers récapitulatifs modifiés : {len(changed_summary)}")

    if not new_responses and not changed_summary:
        print("HDFS est déjà à jour. Aucun transfert nécessaire.")
        return

    container_tmp = run(
        "docker", "compose", "exec", "-T", "namenode",
        "mktemp", "-d", "/tmp/orggraph-sync-XXXXXXXX",
    ).decode("utf-8").strip()

    try:
        with tempfile.TemporaryDirectory(prefix="orggraph-sync-") as tmp:
            payload = Path(tmp) / "payload"
            staged_responses = payload / "responses"
            staged_responses.mkdir(parents=True)

            # Seuls les fichiers nécessaires passent dans le conteneur.
            for path in new_responses:
                shutil.copy2(path, staged_responses / path.name)

            for name in changed_summary:
                shutil.copy2(LOCAL / name, payload / name)

            run(
                "docker", "compose", "cp",
                str(payload),
                f"namenode:{container_tmp}/",
            )

            # Les réponses absentes sont ajoutées sans écrasement.
            for start in range(0, len(new_responses), 100):
                batch = new_responses[start:start + 100]

                sources = [
                    f"{container_tmp}/payload/responses/{path.name}"
                    for path in batch
                ]

                hdfs("-put", *sources, f"{REMOTE}/responses/")

                print(
                    f"Réponses envoyées : "
                    f"{min(start + 100, len(new_responses))}"
                    f"/{len(new_responses)}"
                )

            # Vérifie la taille des nouvelles réponses transférées.
            updated_responses = list_files(f"{REMOTE}/responses")

            for path in new_responses:
                if updated_responses.get(path.name) != path.stat().st_size:
                    raise RuntimeError(
                        f"Vérification échouée : {path.name}"
                    )

            # Le rapport est envoyé en dernier.
            for name in changed_summary:
                hdfs(
                    "-put", "-f",
                    f"{container_tmp}/payload/{name}",
                    f"{REMOTE}/{name}",
                )

                uploaded = hdfs("-cat", f"{REMOTE}/{name}")

                if sha256(uploaded) != sha256((LOCAL / name).read_bytes()):
                    raise RuntimeError(
                        f"Vérification SHA-256 échouée : {name}"
                    )

                print(f"Mis à jour et vérifié : {name}")

        print("\nUPLOAD TERMINÉ")
        print(f"Destination : {REMOTE}")
        print(f"Segments validés : {report['chunks_validated']}")
        print(f"Relations acceptées : {report['accepted_relations']}")

    finally:
        subprocess.run(
            [
                "docker", "compose", "exec", "-T", "namenode",
                "rm", "-rf", "--", container_tmp,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )


if __name__ == "__main__":
    main()