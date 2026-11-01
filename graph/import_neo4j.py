import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Identifiant logique stable, utilisé aussi par l'API,
# graph.analytics et graph.graphrag.
DATASET = "mvp_ff7e5205b7ffa21f_1139"

HDFS_BASE = os.getenv(
    "ORGGRAPH_GOLD_PATH",
    "/data/orggraph/gold/mvp_ff7e5205b7ffa21f_1139",
).rstrip("/")

LABELS = {
    "person": "Person",
    "organization": "Organization",
}

ALLOWED_RELATIONS = {
    "CEO_OF",
    "CHAIRPERSON_OF",
    "BOARD_MEMBER_OF",
    "FOUNDED",
    "EMPLOYED_BY",
    "MEMBER_OF",
    "PARENT_OF",
    "OWNED_BY",
    "PUBLISHED_BY",
}

BATCH_SIZE = 200


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


def read_table(name):
    content = read_hdfs(
        f"{HDFS_BASE}/json/{name}/part-*.json"
    )

    return [
        json.loads(line)
        for line in content.splitlines()
        if line.strip()
    ]


def batches(rows):
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start:start + BATCH_SIZE]


def sync_graph(
    tx,
    nodes_by_label,
    relationships_by_type,
    entity_ids,
    relation_ids,
):
    # Toutes les écritures ci-dessous appartiennent
    # à une seule transaction.

    for label, rows in nodes_by_label.items():
        # label provient exclusivement de LABELS.
        statement = f"""
            UNWIND $rows AS row
            MERGE (n:Entity {{entity_id: row.entity_id}})
            SET n:{label}
            SET n += row.props
        """

        for batch in batches(rows):
            tx.run(statement, rows=batch).consume()

    for relation_type, rows in relationships_by_type.items():
        # Type vérifié avant la transaction.
        statement = f"""
            UNWIND $rows AS row
            MATCH (a:Entity {{entity_id: row.source_id}})
            MATCH (b:Entity {{entity_id: row.target_id}})
            MERGE (a)-[r:{relation_type} {{
                relation_id: row.relation_id
            }}]->(b)
            SET r += row.props
            RETURN count(r) AS processed
        """

        for batch in batches(rows):
            record = tx.run(statement, rows=batch).single()

            if record["processed"] != len(batch):
                raise RuntimeError(
                    "Certaines relations n'ont pas été importées."
                )

    # Synchronise la version active :
    # enlève uniquement ses relations absentes du nouveau Gold.
    removed = tx.run("""
        MATCH (:Entity)-[r]->(:Entity)
        WHERE r.dataset_id = $dataset
          AND (
              r.relation_id IS NULL
              OR NOT (r.relation_id IN $ids)
          )
        WITH collect(r) AS obsolete
        FOREACH (r IN obsolete | DELETE r)
        RETURN size(obsolete) AS removed
    """, dataset=DATASET, ids=relation_ids).single()["removed"]

    # Ne supprime pas les anciens nœuds :
    # retire seulement leur appartenance à ce dataset
    # s'ils ne figurent plus dans sa nouvelle version.
    tx.run("""
        MATCH (n:Entity)
        WHERE n.dataset_id = $dataset
          AND NOT (n.entity_id IN $ids)
        REMOVE n.dataset_id
    """, dataset=DATASET, ids=entity_ids).consume()

    node_count = tx.run("""
        MATCH (n:Entity)
        WHERE n.dataset_id = $dataset
        RETURN count(n) AS total
    """, dataset=DATASET).single()["total"]

    relationship_count = tx.run("""
        MATCH (:Entity)-[r]->(:Entity)
        WHERE r.dataset_id = $dataset
        RETURN count(r) AS total
    """, dataset=DATASET).single()["total"]

    if node_count != len(entity_ids):
        raise RuntimeError(
            f"Entités : {node_count} au lieu de {len(entity_ids)}."
        )

    if relationship_count != len(relation_ids):
        raise RuntimeError(
            f"Relations : {relationship_count} "
            f"au lieu de {len(relation_ids)}."
        )

    # Métadonnées de la version effectivement publiée.
    tx.run("""
        MERGE (d:OrgGraphDataset {dataset_id: $dataset})
        SET d.gold_path = $gold_path,
            d.updated_at = $updated_at,
            d.entity_count = $entity_count,
            d.relation_count = $relation_count
    """,
        dataset=DATASET,
        gold_path=HDFS_BASE,
        updated_at=datetime.now(timezone.utc).isoformat(),
        entity_count=node_count,
        relation_count=relationship_count,
    ).consume()

    return {
        "dataset": DATASET,
        "gold_path": HDFS_BASE,
        "entities": node_count,
        "relations": relationship_count,
        "obsolete_relations_removed": removed,
    }


def main():
    password = os.getenv("NEO4J_PASSWORD", "").strip()

    if not password:
        raise ValueError("Configure NEO4J_PASSWORD dans .env.")

    print(f"Lecture Gold : {HDFS_BASE}", flush=True)

    # La présence du rapport indique que le traitement Spark
    # a atteint sa phase finale.
    spark_report = json.loads(
        read_hdfs(f"{HDFS_BASE}/report/part-*").strip()
    )

    entities = read_table("entities")
    relations = read_table("relations")

    if not entities or not relations:
        raise RuntimeError(
            "Gold vide : mise à jour du graphe refusée."
        )

    if len(entities) != spark_report["entities"]:
        raise RuntimeError(
            "Le nombre d'entités diffère du rapport Spark."
        )

    if (
        len(relations)
        != spark_report["relations_after_deduplication"]
    ):
        raise RuntimeError(
            "Le nombre de relations diffère du rapport Spark."
        )

    entity_ids = [row["entity_id"] for row in entities]
    relation_ids = [row["relation_id"] for row in relations]
    entity_id_set = set(entity_ids)

    if len(entity_id_set) != len(entity_ids):
        raise ValueError("Identifiants d'entités dupliqués.")

    if len(set(relation_ids)) != len(relation_ids):
        raise ValueError("Identifiants de relations dupliqués.")

    nodes_by_label = defaultdict(list)
    relationships_by_type = defaultdict(list)

    for entity in entities:
        entity_type = entity["entity_type"]

        if entity_type not in LABELS:
            raise ValueError(
                f"Type d'entité inconnu : {entity_type}"
            )

        nodes_by_label[LABELS[entity_type]].append({
            "entity_id": entity["entity_id"],
            "props": {
                "entity_type": entity_type,
                "name": entity["name"],
                "name_key": entity["name_key"],
                "name_variants": entity["name_variants"],
                "source_document_ids": (
                    entity["source_document_ids"]
                ),
                "resolution_method": entity["resolution_method"],
                "dataset_id": DATASET,
            },
        })

    for relation in relations:
        kind = relation["relation_type"]

        if kind not in ALLOWED_RELATIONS:
            raise ValueError(
                f"Type de relation inconnu : {kind}"
            )

        if (
            relation["source_id"] not in entity_id_set
            or relation["target_id"] not in entity_id_set
        ):
            raise ValueError("Extrémité de relation absente.")

        records = relation["evidence_records"]

        properties = {
            "role_title": relation.get("role_title"),
            "start_date": relation.get("start_date"),
            "end_date": relation.get("end_date"),
            "temporal_status": relation["temporal_status"],
            "extraction_count": relation["extraction_count"],
            "dataset_id": DATASET,
            "evidence": sorted({
                record["evidence"]
                for record in records
            }),
            "source_urls": sorted({
                record["provenance"]["source_url"]
                for record in records
                if record["provenance"].get("source_url")
            }),
            "chunk_ids": sorted({
                record["provenance"]["chunk_id"]
                for record in records
            }),
            "evidence_records_json": json.dumps(
                records,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        relationships_by_type[kind].append({
            "relation_id": relation["relation_id"],
            "source_id": relation["source_id"],
            "target_id": relation["target_id"],
            "props": properties,
        })

    with GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", password),
    ) as driver:
        driver.verify_connectivity()

        driver.execute_query("""
            CREATE CONSTRAINT orggraph_entity_id IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE n.entity_id IS UNIQUE
        """, database_="neo4j")

        driver.execute_query("""
            CREATE CONSTRAINT orggraph_dataset_id IF NOT EXISTS
            FOR (d:OrgGraphDataset)
            REQUIRE d.dataset_id IS UNIQUE
        """, database_="neo4j")

        with driver.session(database="neo4j") as session:
            report = session.execute_write(
                sync_graph,
                dict(nodes_by_label),
                dict(relationships_by_type),
                entity_ids,
                relation_ids,
            )

    print("\nIMPORT NEO4J VÉRIFIÉ")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()