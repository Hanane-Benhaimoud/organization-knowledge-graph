import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATASET = "orggraph_enriched_v1"
EMBED_MODEL = "intfloat/multilingual-e5-small"
VECTOR_INDEX = "orggraph_enriched_vector"
TEXT_INDEX = "orggraph_enriched_text"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def node_key(identifier):
    return f"{DATASET}|{identifier}"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


@lru_cache(maxsize=1)
def driver():
    password = os.getenv("NEO4J_PASSWORD", "").strip()

    if not password:
        raise ValueError("NEO4J_PASSWORD manque dans .env.")

    connection = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=("neo4j", password),
    )
    connection.verify_connectivity()
    return connection


def query(statement, **parameters):
    records, _, _ = driver().execute_query(
        statement,
        parameters_=parameters,
        database_="neo4j",
        routing_="r",
    )
    return [record.data() for record in records]


def write(statement, **parameters):
    records, _, _ = driver().execute_query(
        statement,
        parameters_=parameters,
        database_="neo4j",
    )
    return [record.data() for record in records]


def state():
    rows = query("""
        MATCH (d:KGDataset {dataset_id: $dataset})
        RETURN properties(d) AS state
    """, dataset=DATASET)

    return rows[0]["state"] if rows else {}


@lru_cache(maxsize=1)
def embedding_model(revision):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)

    model = SentenceTransformer(
        EMBED_MODEL,
        revision=revision,
        device="cpu",
    )

    model.max_seq_length = 512

    if model.get_sentence_embedding_dimension() != 384:
        raise RuntimeError("Dimension d'embedding inattendue.")

    return model