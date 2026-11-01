import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Source : articles déjà chargés dans HDFS.
HDFS_SOURCE = "/data/orggraph/bronze/wikipedia"

# Résultats locaux de cette étape.
OUTPUT_DIR = ROOT / "data" / "silver" / "preparation"
CHUNKS_FILE = OUTPUT_DIR / "wikipedia_chunks.jsonl"
REPORT_FILE = OUTPUT_DIR / "preparation_report.json"

# Limites en caractères, PAS en tokens.
# La taille réelle en tokens sera contrôlée avant les appels LLM.
MAX_CHARS = 4500
OVERLAP_CHARS = 500

# On conserve notamment History, Governance, Management, Career...
# Seules les sections bibliographiques / de navigation sont exclues.
EXCLUDED_HEADINGS = {
    "references",
    "notes",
    "notes and references",
    "notes et references",
    "sources",
    "bibliography",
    "bibliographie",
    "further reading",
    "external links",
    "liens externes",
    "see also",
    "voir aussi",
}


def run(*args):
    """Exécute une commande sans passer par un shell."""
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def normalize_heading(value):
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_text(value):
    """Préserve les paragraphes ; retire seulement les espaces inutiles."""
    value = value.replace("\u00a0", " ")
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in value.splitlines()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_text(text):
    """
    Renvoie (début, fin, texte) dans le texte nettoyé de la section.

    Préférence de coupure :
    paragraphe, fin de phrase, puis espace.
    Les offsets sont des positions en caractères.
    """
    start = 0

    while start < len(text):
        end = min(start + MAX_CHARS, len(text))

        if end < len(text):
            # Évite de couper trop tôt dans le segment.
            lower = start + MAX_CHARS // 2

            for separator in ("\n\n", "\n", ". ", "? ", "! ", " "):
                position = text.rfind(separator, lower, end)
                if position != -1:
                    end = position + len(separator)
                    break

        piece = text[start:end]

        # Les offsets correspondent exactement au texte enregistré.
        left_trim = len(piece) - len(piece.lstrip())
        right_edge = len(piece.rstrip())

        if right_edge > left_trim:
            yield (
                start + left_trim,
                start + right_edge,
                piece.strip(),
            )

        if end == len(text):
            break

        next_start = max(start + 1, end - OVERLAP_CHARS)

        # Évite de commencer au milieu d'un mot.
        while (
            next_start < end
            and next_start > 0
            and not text[next_start - 1].isspace()
        ):
            next_start += 1

        start = next_start


def download_from_hdfs(local_directory):
    """
    HDFS -> dossier temporaire du conteneur -> dossier temporaire local.
    Aucune suppression ni modification des fichiers HDFS.
    """
    container_tmp = run(
        "docker", "compose", "exec", "-T", "namenode",
        "mktemp", "-d", "/tmp/orggraph-chunking-XXXXXXXX",
    ).strip()

    try:
        run(
            "docker", "compose", "exec", "-T", "namenode",
            "hdfs", "dfs", "-get",
            HDFS_SOURCE,
            f"{container_tmp}/wikipedia",
        )

        run(
            "docker", "compose", "cp",
            f"namenode:{container_tmp}/wikipedia",
            str(local_directory),
        )

    finally:
        # Nettoyage limité au dossier temporaire créé par ce script.
        run(
            "docker", "compose", "exec", "-T", "namenode",
            "rm", "-rf", "--", container_tmp,
        ) if False else None

        # rm est nécessaire car le dossier contient la copie des articles.
        run(
            "docker", "compose", "exec", "-T", "namenode",
            "rm", "-rf", "--", container_tmp,
        )

    return local_directory / "wikipedia"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "hdfs_source": HDFS_SOURCE,
        "max_chars": MAX_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "documents_found": 0,
        "documents_processed": 0,
        "chunks_total": 0,
        "excluded_sections": 0,
        "documents_using_full_text_fallback": 0,
        "chunks_per_document": {},
        "skipped": [],
        "errors": [],
    }

    languages = Counter()
    temporary_output = CHUNKS_FILE.with_suffix(".jsonl.tmp")

    with tempfile.TemporaryDirectory(prefix="orggraph-prepare-") as tmp:
        source = download_from_hdfs(Path(tmp))

        # Le rapport de crawl n'est pas un article.
        files = sorted(
            path for path in source.glob("*.json")
            if path.name != "crawl_summary.json"
        )

        report["documents_found"] = len(files)

        if not files:
            raise RuntimeError("Aucun article JSON trouvé dans HDFS.")

        with temporary_output.open("w", encoding="utf-8") as output:
            for path in files:
                try:
                    document = json.loads(
                        path.read_text(encoding="utf-8")
                    )

                    qid = document["wikidata_id"]
                    document_id = document["document_id"]
                    revision_id = document["revision_id"]
                    language = document["language"]
                    source_url = document["source_url"]

                    content = document.get("content") or {}
                    sections = content.get("sections") or []

                    if not sections:
                        sections = [{
                            "heading": "Full text",
                            "text": content.get("full_text", ""),
                        }]
                        report["documents_using_full_text_fallback"] += 1

                    # Préparation complète du document avant son écriture.
                    document_chunks = []

                    for section_index, section in enumerate(sections):
                        heading = str(
                            section.get("heading") or "Introduction"
                        ).strip()

                        if normalize_heading(heading) in EXCLUDED_HEADINGS:
                            report["excluded_sections"] += 1
                            continue

                        text = clean_text(section.get("text") or "")

                        if not text:
                            continue

                        for part_index, (start, end, piece) in enumerate(
                            split_text(text)
                        ):
                            identity = (
                                f"{document_id}|{revision_id}|"
                                f"{section_index}|{part_index}|{piece}"
                            )
                            chunk_id = hashlib.sha256(
                                identity.encode("utf-8")
                            ).hexdigest()

                            document_chunks.append({
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "wikidata_id": qid,
                                "entity_label": document.get("entity_label"),
                                "entity_type": document.get("entity_type"),
                                "seed_categories": document.get(
                                    "seed_categories", []
                                ),
                                "source": "wikipedia",
                                "source_url": source_url,
                                "revision_url": document.get("revision_url"),
                                "revision_id": revision_id,
                                "language": language,
                                "title": document.get("title"),
                                "collected_at": document.get("collected_at"),
                                "section_index": section_index,
                                "section_heading": heading,
                                "part_index": part_index,
                                "char_start": start,
                                "char_end": end,
                                "char_count": len(piece),
                                "text": piece,
                            })

                    if not document_chunks:
                        report["skipped"].append({
                            "file": path.name,
                            "reason": "no_usable_text",
                        })
                        continue

                    for chunk in document_chunks:
                        output.write(
                            json.dumps(chunk, ensure_ascii=False) + "\n"
                        )

                    count = len(document_chunks)
                    report["documents_processed"] += 1
                    report["chunks_total"] += count
                    report["chunks_per_document"][document_id] = count
                    languages[language] += count

                    print(f"{path.name}: {count} segments")

                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append({
                        "file": path.name,
                        "error": str(exc),
                    })

    report["chunks_by_language"] = dict(languages)

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Ne remplace pas un résultat précédent par un corpus incomplet.
    if report["errors"] or report["skipped"]:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Préparation incomplète : consulte preparation_report.json. "
            "Le fichier final de segments n'a pas été remplacé."
        )

    temporary_output.replace(CHUNKS_FILE)

    print("\nPréparation terminée")
    print(f"Documents : {report['documents_processed']}")
    print(f"Segments  : {report['chunks_total']}")
    print(f"Sortie    : {CHUNKS_FILE}")
    print(f"Rapport   : {REPORT_FILE}")


if __name__ == "__main__":
    main()