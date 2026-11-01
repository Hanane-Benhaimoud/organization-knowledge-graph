import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATASET = "mvp_ff7e5205b7ffa21f_1139"

QUERIES = {
    "person_affiliations": {
        "question": (
            "À quelles organisations cette personne est-elle liée, "
            "avec quelles fonctions, périodes et preuves ?"
        ),
        "cypher": """
            MATCH (p:Person)-[r]->(o:Organization)
            WHERE p.name_key = toLower(trim($person))
              AND r.dataset_id = $dataset
              AND type(r) IN [
                  'CEO_OF', 'CHAIRPERSON_OF', 'BOARD_MEMBER_OF',
                  'EMPLOYED_BY', 'MEMBER_OF', 'FOUNDED'
              ]
            RETURN p.entity_id AS person_id,
                   p.name AS person,
                   type(r) AS relation,
                   o.entity_id AS organization_id,
                   o.name AS organization,
                   r.relation_id AS relation_id,
                   r.role_title AS role,
                   r.start_date AS start_date,
                   r.end_date AS end_date,
                   r.temporal_status AS temporal_status,
                   r.evidence AS evidence,
                   r.source_urls AS source_urls,
                   r.chunk_ids AS chunk_ids
            ORDER BY organization, relation, start_date
        """,
    },

    "multiple_affiliations": {
        "question": (
            "Quelles personnes sont liées au plus grand nombre "
            "d'organisations distinctes ?"
        ),
        "cypher": """
            MATCH (p:Person)-[r]->(o:Organization)
            WHERE r.dataset_id = $dataset
              AND type(r) IN [
                  'CEO_OF', 'CHAIRPERSON_OF', 'BOARD_MEMBER_OF',
                  'EMPLOYED_BY', 'MEMBER_OF', 'FOUNDED'
              ]
            WITH p,
                 collect(DISTINCT o.name) AS organizations,
                 count(DISTINCT o.entity_id) AS organization_count
            WHERE organization_count > 1
            RETURN p.entity_id AS person_id,
                   p.name AS person,
                   organizations,
                   organization_count
            ORDER BY organization_count DESC, person
            LIMIT 20
        """,
    },

    "shared_leaders": {
        "question": (
            "Quelles paires d'organisations partagent des personnes "
            "ayant une fonction de direction ou de gouvernance ?"
        ),
        "cypher": """
            MATCH (a:Organization)<-[r1]-(p:Person)-[r2]->(b:Organization)
            WHERE a.entity_id < b.entity_id
              AND r1.dataset_id = $dataset
              AND r2.dataset_id = $dataset
              AND type(r1) IN [
                  'CEO_OF', 'CHAIRPERSON_OF', 'BOARD_MEMBER_OF'
              ]
              AND type(r2) IN [
                  'CEO_OF', 'CHAIRPERSON_OF', 'BOARD_MEMBER_OF'
              ]
            WITH a, b,
                 collect(DISTINCT p.name) AS shared_people,
                 count(DISTINCT p.entity_id) AS shared_people_count
            RETURN a.entity_id AS organization_a_id,
                   a.name AS organization_a,
                   b.entity_id AS organization_b_id,
                   b.name AS organization_b,
                   shared_people,
                   shared_people_count
            ORDER BY shared_people_count DESC,
                     organization_a, organization_b
            LIMIT 30
        """,
    },

    "connected_entities": {
        "question": (
            "Quelles entités ont le plus de voisins distincts "
            "dans ce graphe ?"
        ),
        "cypher": """
            MATCH (n:Entity)-[r]-(neighbor:Entity)
            WHERE r.dataset_id = $dataset
              AND n.entity_id <> neighbor.entity_id
            WITH n,
                 count(DISTINCT neighbor.entity_id) AS neighbor_count
            RETURN n.entity_id AS entity_id,
                   n.name AS name,
                   n.entity_type AS entity_type,
                   neighbor_count
            ORDER BY neighbor_count DESC, name
            LIMIT 20
        """,
    },

    "subsidiary_chains": {
        "question": (
            "Quelles organisations sont accessibles depuis cette "
            "organisation par un à trois liens de filiation ?"
        ),
        "cypher": """
            MATCH path =
                (parent:Organization)-[:PARENT_OF*1..3]->
                (child:Organization)
            WHERE parent.name_key = toLower(trim($organization))
              AND parent.entity_id <> child.entity_id
              AND all(
                  rel IN relationships(path)
                  WHERE rel.dataset_id = $dataset
              )
            RETURN parent.name AS parent,
                   child.name AS subsidiary,
                   length(path) AS hops,
                   [n IN nodes(path) | n.name] AS chain,
                   [rel IN relationships(path) |
                       {
                           relation_id: rel.relation_id,
                           temporal_status: rel.temporal_status,
                           evidence: rel.evidence,
                           source_urls: rel.source_urls
                       }
                   ] AS supporting_relations
            ORDER BY hops, subsidiary, chain
            LIMIT 50
        """,
    },
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--person",
        default="Thomas Schäfer",
    )
    parser.add_argument(
        "--organization",
        default="Google",
    )
    parser.add_argument(
        "--query",
        choices=["all", *QUERIES],
        default="all",
    )

    args = parser.parse_args()

    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if not password:
        raise ValueError("Configure NEO4J_PASSWORD dans .env.")

    parameters = {
        "dataset": DATASET,
        "person": args.person,
        "organization": args.organization,
    }

    selected = (
        QUERIES
        if args.query == "all"
        else {args.query: QUERIES[args.query]}
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = ROOT / "data" / "analytics" / DATASET / timestamp
    output.mkdir(parents=True, exist_ok=False)

    summary = []

    with GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", password),
    ) as driver:
        driver.verify_connectivity()

        for name, definition in selected.items():
            start = perf_counter()

            records, _, _ = driver.execute_query(
                definition["cypher"],
                parameters_=parameters,
                database_="neo4j",
                routing_="r",
            )

            rows = [record.data() for record in records]
            elapsed_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )

            result = {
                "query_id": name,
                "question": definition["question"],
                "method": "cypher",
                "dataset_id": DATASET,
                "parameters": parameters,
                "cypher": definition["cypher"],
                "measured_client_latency_ms": elapsed_ms,
                "row_count": len(rows),
                "rows": rows,
            }

            (output / f"{name}.json").write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary.append({
                "query_id": name,
                "row_count": len(rows),
                "measured_client_latency_ms": elapsed_ms,
            })

            print(
                f"\n{name} : {len(rows)} résultats "
                f"| {elapsed_ms} ms"
            )

            print(json.dumps(
                rows[:3],
                ensure_ascii=False,
                indent=2,
            ))

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRésultats sauvegardés : {output}")


if __name__ == "__main__":
    main()