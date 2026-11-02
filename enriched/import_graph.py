import argparse
import json
import subprocess
from datetime import datetime, timezone

from enriched.common import (
    DATASET,
    ROOT,
    driver,
    node_key,
    save_json,
    write,
)


TABLES = {
    "documents": ("KGDocument", "document_version_id"),
    "chunks": ("KGChunk", "chunk_id"),
    "entities": ("KGEntity", "entity_id"),
    "relations": ("KGStatement", "relation_id"),
    "evidence": ("KGEvidence", "evidence_id"),
}


def read_hdfs(pattern):
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "namenode",
            "hdfs", "dfs", "-cat", pattern,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def read_table(base, name):
    return [
        json.loads(line)
        for line in read_hdfs(
            f"{base}/json/{name}/part-*.json"
        ).splitlines()
        if line.strip()
    ]


def batches(rows, size=200):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def import_transaction(tx, tables, gold, report):
    # Ne touche qu'aux nœuds de la version enrichie.
    tx.run("""
        MATCH (n:KGNode {dataset_id: $dataset})
        DETACH DELETE n
    """, dataset=DATASET).consume()

    for table, (label, identifier) in TABLES.items():
        rows = []

        for item in tables[table]:
            props = dict(item)
            props["id"] = item[identifier]
            props["dataset_id"] = DATASET

            rows.append({
                "key": node_key(item[identifier]),
                "props": props,
            })

        statement = f"""
            UNWIND $rows AS row
            CREATE (n:KGNode:{label} {{key: row.key}})
            SET n += row.props
        """

        for batch in batches(rows):
            tx.run(statement, rows=batch).consume()

    links = [
        (
            "chunks",
            "chunk_id",
            "document_version_id",
            "IN_DOCUMENT",
        ),
        (
            "evidence",
            "evidence_id",
            "chunk_id",
            "FROM_CHUNK",
        ),
    ]

    for table, source_field, target_field, relation in links:
        rows = [
            {
                "source": node_key(item[source_field]),
                "target": node_key(item[target_field]),
            }
            for item in tables[table]
        ]

        for batch in batches(rows):
            record = tx.run(f"""
                UNWIND $rows AS row
                MATCH (a:KGNode {{key: row.source}})
                MATCH (b:KGNode {{key: row.target}})
                CREATE (a)-[:{relation}]->(b)
                RETURN count(*) AS processed
            """, rows=batch).single()

            if record["processed"] != len(batch):
                raise RuntimeError(f"Liens manquants : {relation}")

    # Une connaissance devient un nœud KGStatement :
    # il peut ainsi être relié explicitement à plusieurs preuves.
    for relation, target_field in (
        ("SUBJECT", "subject_id"),
        ("OBJECT", "object_id"),
    ):
        rows = [
            {
                "statement": node_key(item["relation_id"]),
                "entity": node_key(item[target_field]),
            }
            for item in tables["relations"]
        ]

        for batch in batches(rows):
            record = tx.run(f"""
                UNWIND $rows AS row
                MATCH (s:KGStatement {{key: row.statement}})
                MATCH (e:KGEntity {{key: row.entity}})
                CREATE (s)-[:{relation}]->(e)
                RETURN count(*) AS processed
            """, rows=batch).single()

            if record["processed"] != len(batch):
                raise RuntimeError(f"Extrémités manquantes : {relation}")

    rows = [
        {
            "statement": node_key(item["relation_id"]),
            "evidence": node_key(item["evidence_id"]),
        }
        for item in tables["evidence"]
    ]

    for batch in batches(rows):
        record = tx.run("""
            UNWIND $rows AS row
            MATCH (s:KGStatement {key: row.statement})
            MATCH (e:KGEvidence {key: row.evidence})
            CREATE (s)-[:SUPPORTED_BY]->(e)
            RETURN count(*) AS processed
        """, rows=batch).single()

        if record["processed"] != len(batch):
            raise RuntimeError("Preuves non rattachées.")

    rows = [
        {
            "chunk": node_key(item["chunk_id"]),
            "entity": node_key(item["entity_id"]),
            "id": item["mention_id"],
            "surfaces": item["surface_forms"],
        }
        for item in tables["mentions"]
    ]

    for batch in batches(rows):
        record = tx.run("""
            UNWIND $rows AS row
            MATCH (c:KGChunk {key: row.chunk})
            MATCH (e:KGEntity {key: row.entity})
            CREATE (c)-[:MENTIONS {
                id: row.id,
                surface_forms: row.surfaces
            }]->(e)
            RETURN count(*) AS processed
        """, rows=batch).single()

        if record["processed"] != len(batch):
            raise RuntimeError("Mentions non rattachées.")

    for table, (label, _) in TABLES.items():
        count = tx.run(f"""
            MATCH (n:{label} {{dataset_id: $dataset}})
            RETURN count(n) AS total
        """, dataset=DATASET).single()["total"]

        if count != len(tables[table]):
            raise RuntimeError(f"Comptage incohérent : {table}")

    tx.run("""
        MERGE (d:KGDataset {dataset_id: $dataset})
        SET d.gold_path = $gold,
            d.phase = 'graph_loaded',
            d.updated_at = $updated,
            d.embedding_signature = null,
            d.embedding_revision = null
    """,
        dataset=DATASET,
        gold=gold,
        updated=datetime.now(timezone.utc).isoformat(),
    ).consume()

    return report["tables"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()

    gold = args.gold.rstrip("/")

    report = json.loads(
        read_hdfs(f"{gold}/report/part-*").strip()
    )

    if report.get("all_integrity_checks_passed") is not True:
        raise RuntimeError("Le rapport Spark ne valide pas l'intégrité.")

    tables = {
        name: read_table(gold, name)
        for name in [*TABLES, "mentions"]
    }

    for name, rows in tables.items():
        if len(rows) != report["tables"][name]:
            raise RuntimeError(f"Nombre de lignes incorrect : {name}")

    for name, (_, identifier) in TABLES.items():
        identifiers = [row[identifier] for row in tables[name]]

        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError(f"Identifiants dupliqués : {name}")

    write("""
        CREATE CONSTRAINT orggraph_kg_key IF NOT EXISTS
        FOR (n:KGNode)
        REQUIRE n.key IS UNIQUE
    """)

    write("""
        CREATE CONSTRAINT orggraph_kg_dataset IF NOT EXISTS
        FOR (n:KGDataset)
        REQUIRE n.dataset_id IS UNIQUE
    """)

    with driver().session(database="neo4j") as session:
        counts = session.execute_write(
            import_transaction,
            tables,
            gold,
            report,
        )

    save_json(
        ROOT / "data" / "pipeline" / "enriched_graph.json",
        {"dataset": DATASET, "gold": gold, "tables": counts},
    )

    print("\nIMPORT ENRICHI TERMINÉ")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("Phase : graph_loaded")
    print("Le MVP est conservé.")


if __name__ == "__main__":
    main()