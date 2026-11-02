import json
import os

from huggingface_hub import model_info

from enriched.common import (
    DATASET,
    EMBED_MODEL,
    ROOT,
    TEXT_INDEX,
    VECTOR_INDEX,
    digest,
    embedding_model,
    node_key,
    query,
    save_json,
    state,
    write,
)


def main():
    current = state()

    if current.get("phase") not in ("graph_loaded", "ready"):
        raise RuntimeError("Importe d'abord le graphe enrichi.")

    # Fixe le commit réel du modèle pour rendre le cache traçable.
    revision = (
        os.getenv("EMBEDDING_REVISION", "").strip()
        or model_info(EMBED_MODEL).sha
    )

    model = embedding_model(revision)
    tokenizer = model.tokenizer

    if not tokenizer.is_fast:
        raise RuntimeError(
            "Un tokenizer rapide avec positions de caractères est requis."
        )

    signature = f"{EMBED_MODEL}@{revision}:window400:stride320:v1"

    chunks = query("""
        MATCH (c:KGChunk {dataset_id: $dataset})
        RETURN c.key AS key,
               c.id AS id,
               c.text AS text,
               c.title AS title,
               c.section_heading AS section
        ORDER BY c.id
    """, dataset=DATASET)

    cache = ROOT / "data" / "embeddings"
    units = []

    for chunk in chunks:
        tokens = tokenizer(
            chunk["text"],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        offsets = tokens["offset_mapping"]

        heading = f"{chunk['title']} / {chunk['section'] or ''}"

        heading = tokenizer.decode(
            tokenizer.encode(
                heading,
                add_special_tokens=False,
            )[:40],
            skip_special_tokens=True,
        )

        for position in range(0, len(offsets), 320):
            last = min(position + 400, len(offsets))
            start_char = offsets[position][0]
            end_char = offsets[last - 1][1]

            text = chunk["text"][start_char:end_char]

            embedding_input = f"passage: {heading}\n{text}"

            if len(tokenizer.encode(embedding_input)) > 512:
                raise RuntimeError(
                    "Une fenêtre dépasse la limite du modèle."
                )

            cache_id = digest({
                "signature": signature,
                "text": embedding_input,
            })

            identifier = "passage:" + digest({
                "chunk": chunk["id"],
                "start": start_char,
                "end": end_char,
                "signature": signature,
                "text": text,
            })

            units.append({
                "key": node_key(identifier),
                "id": identifier,
                "chunk_key": chunk["key"],
                "text": text,
                "start_char": start_char,
                "end_char": end_char,
                "embedding_input": embedding_input,
                "cache_path": cache / f"{cache_id}.json",
            })

            if last == len(offsets):
                break

    pending = [
        unit for unit in units
        if not unit["cache_path"].exists()
    ]

    print(f"Segments : {len(chunks)}", flush=True)
    print(f"Fenêtres : {len(units)}", flush=True)
    print(f"Embeddings à calculer : {len(pending)}", flush=True)

    for start in range(0, len(pending), 16):
        batch = pending[start:start + 16]

        vectors = model.encode(
            [unit["embedding_input"] for unit in batch],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for unit, vector in zip(batch, vectors):
            save_json(unit["cache_path"], {
                "signature": signature,
                "embedding": vector.tolist(),
            })

        print(
            f"Calculés : {min(start + 16, len(pending))}"
            f"/{len(pending)}",
            flush=True,
        )

    # Retire les fenêtres d'un éventuel ancien modèle.
    # Le cache disque reste disponible.
    write("""
        MATCH (p:KGPassage {dataset_id: $dataset})
        WHERE p.model_signature <> $signature
        DETACH DELETE p
    """, dataset=DATASET, signature=signature)

    for start in range(0, len(units), 100):
        rows = []

        for unit in units[start:start + 100]:
            cached = json.loads(
                unit["cache_path"].read_text(encoding="utf-8")
            )

            vector = cached["embedding"]

            if cached["signature"] != signature or len(vector) != 384:
                raise RuntimeError("Cache d'embedding incohérent.")

            rows.append({
                "key": unit["key"],
                "id": unit["id"],
                "chunk_key": unit["chunk_key"],
                "text": unit["text"],
                "start_char": unit["start_char"],
                "end_char": unit["end_char"],
                "embedding": vector,
            })

        result = write("""
            UNWIND $rows AS row
            MATCH (c:KGChunk {key: row.chunk_key})
            MERGE (p:KGNode:KGPassage {key: row.key})
            SET p.id = row.id,
                p.dataset_id = $dataset,
                p.text = row.text,
                p.start_char = row.start_char,
                p.end_char = row.end_char,
                p.embedding = row.embedding,
                p.model_signature = $signature
            MERGE (p)-[:WINDOW_OF]->(c)
            RETURN count(*) AS processed
        """, rows=rows, dataset=DATASET, signature=signature)

        if result[0]["processed"] != len(rows):
            raise RuntimeError("Une fenêtre n'a pas été importée.")

    write(f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (p:KGPassage) ON (p.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)

    write(f"""
        CREATE FULLTEXT INDEX {TEXT_INDEX} IF NOT EXISTS
        FOR (p:KGPassage) ON EACH [p.text]
    """)

    write("CALL db.awaitIndexes(300)")

    count = query("""
        MATCH (p:KGPassage {dataset_id: $dataset})
        WHERE p.model_signature = $signature
        RETURN count(p) AS total
    """, dataset=DATASET, signature=signature)[0]["total"]

    if count != len(units):
        raise RuntimeError("Nombre de fenêtres incohérent.")

    write("""
        MATCH (d:KGDataset {dataset_id: $dataset})
        SET d.phase = 'ready',
            d.embedding_signature = $signature,
            d.embedding_revision = $revision,
            d.passage_count = $count
    """,
        dataset=DATASET,
        signature=signature,
        revision=revision,
        count=count,
    )

    print("\nEMBEDDINGS TERMINÉS")
    print(f"Modèle : {EMBED_MODEL}")
    print(f"Révision : {revision}")
    print(f"Dimension : 384")
    print(f"Fenêtres indexées : {count}")
    print("Phase : ready")


if __name__ == "__main__":
    main()