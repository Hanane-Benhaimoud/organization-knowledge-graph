"""
OrgGraph — interface principale de la version enrichie.

Lancement depuis la racine du projet :
    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

import json
import logging
import re
import textwrap

from itertools import islice
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from groq import APIError, APIStatusError, RateLimitError
from neo4j import READ_ACCESS, unit_of_work
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field

from enriched.common import DATASET, driver, query, state
from enriched.rag import ask


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "api" / "static" / "index.html"

MAX_ROWS = 500

logger = logging.getLogger(__name__)

app = FastAPI(title="OrgGraph — Version enrichie")


# ---------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------

@app.exception_handler(Neo4jError)
async def neo4j_error(request, exc):
    logger.exception("Erreur Neo4j")

    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Neo4j est inaccessible ou une requête a échoué. "
                "Consulte le terminal de l'API."
            )
        },
    )


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.exception("Erreur API")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne. Consulte le terminal de l'API."
        },
    )


# ---------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    if not INDEX.is_file():
        raise HTTPException(
            status_code=404,
            detail="Fichier api/static/index.html introuvable.",
        )

    return FileResponse(
        INDEX,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/version", include_in_schema=False)
def version():
    """Permet de vérifier quel serveur et quel fichier HTML sont servis."""
    html = INDEX.read_text(encoding="utf-8") if INDEX.is_file() else ""

    return {
        "version": "enriched",
        "dataset": DATASET,
        "index_path": str(INDEX),
        "index_is_enriched": "enriched-v1" in html,
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(
        content=None,
        status_code=204,
    )


# ---------------------------------------------------------------------
# État et catalogue
# ---------------------------------------------------------------------

@app.get("/api/status")
def status():
    counts = query("""
        MATCH (n:KGNode {dataset_id: $dataset})
        UNWIND labels(n) AS label
        WITH label, count(*) AS total
        WHERE label <> 'KGNode'
        RETURN label, total
        ORDER BY label
    """, dataset=DATASET)

    return {
        "version": "enriched",
        "dataset": DATASET,
        "state": state(),
        "counts": {
            item["label"]: item["total"]
            for item in counts
        },
    }


@app.get("/api/entities")
def entities():
    return query("""
        MATCH (e:KGEntity {dataset_id: $dataset})
        RETURN e.key AS key,
               e.name AS name,
               e.entity_type AS type,
               e.qid AS qid,
               e.resolution_method AS resolution,
               e.name_variants AS variants
        ORDER BY name, key
    """, dataset=DATASET)


# ---------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------

@app.get("/api/graph")
def graph(
    entity_key: str | None = None,
    limit: int = Query(default=120, ge=1, le=400),
):
    rows = query("""
        MATCH (a:KGEntity)<-[:SUBJECT]-
              (s:KGStatement)-[:OBJECT]->(b:KGEntity)
        WHERE s.dataset_id = $dataset
          AND a.dataset_id = $dataset
          AND b.dataset_id = $dataset
          AND (
              $key IS NULL
              OR a.key = $key
              OR b.key = $key
          )
        RETURN a.key AS source_id,
               a.name AS subject,
               a.entity_type AS source_type,
               a.qid AS source_qid,
               a.resolution_method AS source_resolution,
               b.key AS target_id,
               b.name AS object,
               b.entity_type AS target_type,
               b.qid AS target_qid,
               b.resolution_method AS target_resolution,
               s.key AS statement_key,
               s.relation_type AS relation,
               s.role_title AS role,
               s.start_date AS start_date,
               s.end_date AS end_date,
               s.temporal_status AS status,
               s.supporting_document_count AS documents
        ORDER BY statement_key
        LIMIT $limit
    """, dataset=DATASET, key=entity_key, limit=limit + 1)

    truncated = len(rows) > limit
    rows = rows[:limit]

    nodes = {}
    edges = []

    for row in rows:
        nodes[row["source_id"]] = {
            "id": row["source_id"],
            "label": row["subject"],
            "type": row["source_type"],
            "qid": row["source_qid"],
            "resolution": row["source_resolution"],
        }

        nodes[row["target_id"]] = {
            "id": row["target_id"],
            "label": row["object"],
            "type": row["target_type"],
            "qid": row["target_qid"],
            "resolution": row["target_resolution"],
        }

        edges.append({
            "id": row["statement_key"],
            "from": row["source_id"],
            "to": row["target_id"],
            "label": row["relation"],
            "subject": row["subject"],
            "object": row["object"],
            "role": row["role"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "status": row["status"],
            "documents": row["documents"],
        })

    if entity_key and not nodes:
        isolated = query("""
            MATCH (e:KGEntity {
                dataset_id: $dataset,
                key: $key
            })
            RETURN e.key AS id,
                   e.name AS label,
                   e.entity_type AS type,
                   e.qid AS qid,
                   e.resolution_method AS resolution
        """, dataset=DATASET, key=entity_key)

        for row in isolated:
            nodes[row["id"]] = row

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "truncated": truncated,
    }


@app.get("/api/evidence")
def evidence(statement_key: str):
    return query("""
        MATCH (s:KGStatement {
            dataset_id: $dataset,
            key: $key
        })-[:SUPPORTED_BY]->(e:KGEvidence)
        -[:FROM_CHUNK]->(c:KGChunk)
        -[:IN_DOCUMENT]->(d:KGDocument)
        RETURN e.quote AS quote,
               coalesce(e.revision_url, d.revision_url)
                   AS revision_url,
               coalesce(e.source_url, d.source_url)
                   AS source_url,
               d.title AS title,
               c.section_heading AS section,
               c.chunk_id AS chunk_id
        ORDER BY e.id
        LIMIT 30
    """, dataset=DATASET, key=statement_key)


# ---------------------------------------------------------------------
# Exemples analytics
# ---------------------------------------------------------------------

EXAMPLES = [
    {
        "id": "affiliations",
        "question": "Organisations liées à Thomas Schäfer",
        "cypher": """
            MATCH (p:KGEntity)<-[:SUBJECT]-
                  (s:KGStatement)-[:OBJECT]->(o:KGEntity)
            WHERE s.dataset_id = $dataset
              AND p.name = 'Thomas Schäfer'
              AND o.entity_type = 'organization'
            RETURN p.name AS personne,
                   s.relation_type AS relation,
                   o.name AS organisation,
                   s.role_title AS fonction,
                   s.start_date AS debut,
                   s.end_date AS fin,
                   s.temporal_status AS statut
            ORDER BY organisation, relation
        """,
    },
    {
        "id": "leaders",
        "question": "Dirigeants mentionnés dans les sources",
        "cypher": """
            MATCH (p:KGEntity)<-[:SUBJECT]-
                  (s:KGStatement)-[:OBJECT]->(o:KGEntity)
            WHERE s.dataset_id = $dataset
              AND p.entity_type = 'person'
              AND o.entity_type = 'organization'
              AND s.relation_type IN [
                  'CEO_OF',
                  'CHAIRPERSON_OF',
                  'BOARD_MEMBER_OF'
              ]
            RETURN p.name AS personne,
                   o.name AS organisation,
                   s.relation_type AS relation,
                   s.temporal_status AS statut,
                   s.start_date AS debut,
                   s.end_date AS fin
            ORDER BY organisation, personne
            LIMIT 100
        """,
    },
    {
        "id": "multiple_organizations",
        "question": "Personnes liées à plusieurs organisations",
        "cypher": """
            MATCH (p:KGEntity)<-[:SUBJECT]-
                  (s:KGStatement)-[:OBJECT]->(o:KGEntity)
            WHERE s.dataset_id = $dataset
              AND p.entity_type = 'person'
              AND o.entity_type = 'organization'
            WITH p,
                 count(DISTINCT o) AS nombre,
                 collect(DISTINCT o.name) AS organisations
            WHERE nombre > 1
            RETURN p.name AS personne,
                   nombre,
                   organisations
            ORDER BY nombre DESC, personne
            LIMIT 30
        """,
    },
    {
        "id": "relation_types",
        "question": "Répartition des types de relations",
        "cypher": """
            MATCH (s:KGStatement {dataset_id: $dataset})
            RETURN s.relation_type AS relation,
                   count(*) AS nombre
            ORDER BY nombre DESC
        """,
    },
    {
        "id": "evidence",
        "question": "Relations et preuves documentaires",
        "cypher": """
            MATCH (a:KGEntity)<-[:SUBJECT]-
                  (s:KGStatement)-[:OBJECT]->(b:KGEntity),
                  (s)-[:SUPPORTED_BY]->(e:KGEvidence)
                  -[:FROM_CHUNK]->(c:KGChunk)
                  -[:IN_DOCUMENT]->(d:KGDocument)
            WHERE s.dataset_id = $dataset
            RETURN a.name AS sujet,
                   s.relation_type AS relation,
                   b.name AS objet,
                   e.quote AS preuve,
                   d.title AS document,
                   d.revision_url AS source_url
            ORDER BY sujet, relation, objet
            LIMIT 50
        """,
    },
]


@app.get("/api/analytics")
def analytics_examples():
    return [
        {
            **item,
            "cypher": textwrap.dedent(item["cypher"]).strip(),
        }
        for item in EXAMPLES
    ]


# ---------------------------------------------------------------------
# Console Cypher locale
# ---------------------------------------------------------------------

class CypherInput(BaseModel):
    cypher: str = Field(min_length=1, max_length=20000)


BLOCKED = re.compile(
    r"\b(CALL|LOAD|USE|EXPLAIN|PROFILE)\b"
    r"|//|/\*|\*/|;",
    flags=re.IGNORECASE,
)


def run_cypher(statement):
    statement = statement.strip().rstrip(";").strip()

    if not re.match(
        r"^(MATCH|OPTIONAL\s+MATCH|WITH)\b",
        statement,
        flags=re.IGNORECASE,
    ):
        raise HTTPException(
            422,
            "Commence la requête par MATCH, OPTIONAL MATCH ou WITH.",
        )

    if BLOCKED.search(statement):
        raise HTTPException(
            403,
            "Une seule requête de consultation est acceptée, "
            "sans CALL, LOAD, USE ni commentaire.",
        )

    parameters = {"dataset": DATASET}

    @unit_of_work(timeout=15)
    def work(tx):
        summary = tx.run(
            "EXPLAIN " + statement,
            parameters,
        ).consume()

        if summary.query_type != "r":
            raise HTTPException(
                403,
                "Les requêtes d'écriture ne sont pas acceptées.",
            )

        result = tx.run(statement, parameters)

        return [
            record.data()
            for record in islice(result, MAX_ROWS + 1)
        ]

    started = perf_counter()

    try:
        with driver().session(
            database="neo4j",
            default_access_mode=READ_ACCESS,
        ) as session:
            rows = session.execute_read(work)

    except Neo4jError as exc:
        raise HTTPException(400, exc.message) from exc

    truncated = len(rows) > MAX_ROWS

    rows = json.loads(
        json.dumps(
            rows[:MAX_ROWS],
            ensure_ascii=False,
            default=str,
        )
    )

    return {
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "latency_ms": round(
            (perf_counter() - started) * 1000,
            2,
        ),
        "dataset": DATASET,
    }


@app.post("/api/cypher")
def cypher_endpoint(body: CypherInput):
    return run_cypher(body.cypher)


# ---------------------------------------------------------------------
# GraphRAG enrichi
# ---------------------------------------------------------------------

class RagInput(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    entity: str | None = Field(default=None, max_length=300)
    retrieve_only: bool = False


@app.post("/api/graphrag")
def graphrag(body: RagInput):
    started = perf_counter()

    try:
        result = dict(
            ask(
                body.question,
                body.entity or None,
                body.retrieve_only,
            )
        )

    except RateLimitError as exc:
        delay = exc.response.headers.get("retry-after")

        raise HTTPException(
            429,
            f"Quota Groq atteint. Retry-After : {delay}",
        ) from exc

    except APIStatusError as exc:
        raise HTTPException(
            502,
            f"Groq a refusé la requête : HTTP {exc.status_code}.",
        ) from exc

    except APIError as exc:
        raise HTTPException(
            502,
            "Groq est momentanément inaccessible.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    result["latency_ms"] = round(
        (perf_counter() - started) * 1000,
        2,
    )

    # Filet de sécurité : une ancienne interface en cache lisait
    # result.entities.join(...) et plantait si le champ était absent.
    context = result.get("context") or {}
    result.setdefault(
        "entities",
        list(context.get("recognized_entities") or []),
    )

    return result