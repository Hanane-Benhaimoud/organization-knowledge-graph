"""API OrgGraph : expose Neo4j (analytics), GraphRAG (Groq) et la comparaison.
Lancer depuis la racine du projet :  uvicorn api.main:app --reload
"""
import hashlib
import textwrap
import traceback
from itertools import islice
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from groq import APIError, APIStatusError, Groq, RateLimitError
from neo4j import READ_ACCESS, GraphDatabase, unit_of_work
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel

import re
from graph.analytics import DATASET, QUERIES      # charge aussi le .env
from graph.graphrag import MODEL, SYSTEM_PROMPT, normalize

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "graphrag" / DATASET      # même cache que graphrag.py
BENCH = ROOT / "data" / "benchmark" / "benchmark.json"
STATE = {}


@asynccontextmanager
async def lifespan(app):
    pw = os.getenv("NEO4J_PASSWORD", "").strip()
    if not pw:
        raise ValueError("Configure NEO4J_PASSWORD dans .env.")
    STATE["driver"] = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pw))
    STATE["driver"].verify_connectivity()
    yield
    STATE["driver"].close()


app = FastAPI(title="OrgGraph", lifespan=lifespan)


@app.exception_handler(Exception)
async def any_error(request, exc):
    """Au lieu de « Internal Server Error », renvoie la vraie cause à l'interface."""
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__} : {exc}"})


def cypher(statement, **params):
    records, _, _ = STATE["driver"].execute_query(
        statement, parameters_=params, database_="neo4j", routing_="r")
    return [r.data() for r in records]


EDGE_RETURN = """
    RETURN startNode(r).entity_id AS source_id, startNode(r).name AS subject,
           startNode(r).entity_type AS source_type, type(r) AS relation,
           endNode(r).entity_id AS target_id, endNode(r).name AS object,
           endNode(r).entity_type AS target_type, r.relation_id AS relation_id,
           r.role_title AS role_title, r.start_date AS start_date,
           r.end_date AS end_date, r.temporal_status AS temporal_status,
           r.evidence_records_json AS evidence_records_json"""


def to_graph(rows):
    nodes, edges = {}, []
    for r in rows:
        nodes[r["source_id"]] = {"id": r["source_id"], "label": r["subject"], "type": r["source_type"]}
        nodes[r["target_id"]] = {"id": r["target_id"], "label": r["object"], "type": r["target_type"]}
        edges.append({"id": r["relation_id"], "from": r["source_id"], "to": r["target_id"],
                      "label": r["relation"], "role": r.get("role_title"),
                      "status": r.get("temporal_status")})
    return {"nodes": list(nodes.values()), "edges": edges}


# ---------- Exploration ----------
@app.get("/api/entities")
def entities():
    return cypher("""
        MATCH (n:Entity)-[r]-(:Entity) WHERE r.dataset_id = $d
        RETURN DISTINCT n.entity_id AS id, n.name AS name, n.entity_type AS type
        ORDER BY name""", d=DATASET)


@app.get("/api/graph")
def graph(entity: str | None = None, limit: int = 150):
    limit = max(1, min(limit, 400))
    if entity:
        rows = cypher("""
            MATCH (n:Entity)-[r]-(:Entity)
            WHERE n.name_key = toLower(trim($e)) AND r.dataset_id = $d
            WITH DISTINCT r LIMIT $l""" + EDGE_RETURN, e=entity, d=DATASET, l=limit)
    else:
        rows = cypher("""
            MATCH (:Entity)-[r]->(:Entity) WHERE r.dataset_id = $d
            WITH r LIMIT $l""" + EDGE_RETURN, d=DATASET, l=limit)
    for r in rows:
        r.pop("evidence_records_json", None)
    return to_graph(rows)


# ---------- Graph analytics : Cypher libre (lecture seule) ----------
MAX_ROWS = 500
FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|GRANT|DENY|REVOKE|ALTER|TERMINATE)\b"
    r"|apoc\.(create|merge|refactor|periodic|cypher|load|export|trigger)|dbms\.|;\s*\S", re.I)


class CypherIn(BaseModel):
    cypher: str


def fill(cy):
    """Les exemples du projet utilisent $person / $organization : on met des valeurs réelles."""
    return cy.replace("$person", "'Thomas Schäfer'").replace("$organization", "'Google'")


def run_cypher(cy):
    cy = cy.strip().rstrip(";").strip()
    if not cy:
        raise HTTPException(422, "Requête vide.")
    if FORBIDDEN.search(cy):
        raise HTTPException(403, "Lecture seule : écriture, import et multi-requêtes sont interdits.")
    params = {"dataset": DATASET}

    @unit_of_work(timeout=15)
    def work(tx):
        return [r.data() for r in islice(tx.run(cy, params), MAX_ROWS + 1)]

    start = perf_counter()
    try:
        with STATE["driver"].session(database="neo4j", default_access_mode=READ_ACCESS) as session:
            rows = session.execute_read(work)
    except Neo4jError as exc:
        raise HTTPException(400, exc.message)
    latency = round((perf_counter() - start) * 1000, 2)
    truncated = len(rows) > MAX_ROWS
    rows = json.loads(json.dumps(rows[:MAX_ROWS], default=str))
    return {"cypher": cy, "parameters": params, "rows": rows, "row_count": len(rows),
            "truncated": truncated, "latency_ms": latency}


@app.post("/api/cypher")
def cypher_endpoint(q: CypherIn):
    return run_cypher(q.cypher)


@app.get("/api/analytics")      # exemples de questions proposés dans l'interface
def analytics_examples():
    return [{"id": k, "question": v["question"], "cypher": fill(textwrap.dedent(v["cypher"]).strip())}
            for k, v in QUERIES.items()]


# ---------- GraphRAG ----------
class RagIn(BaseModel):
    question: str
    entities: list[str] = []
    hops: int = 1
    max_relations: int = 40


def run_graphrag(q: RagIn):
    if q.hops not in (1, 2) or not 1 <= q.max_relations <= 200:
        raise HTTPException(422, "hops ∈ {1,2} et 1 ≤ max_relations ≤ 200.")
    start = perf_counter()
    catalog = cypher("""
        MATCH (n:Entity)-[r]-(:Entity) WHERE r.dataset_id = $dataset
        RETURN DISTINCT n.entity_id AS entity_id, n.name AS name,
               n.name_variants AS variants, n.entity_type AS entity_type
        ORDER BY name, entity_id""", dataset=DATASET)

    key_q = " " + normalize(q.question) + " "
    requested = {normalize(v) for v in q.entities}
    cands = []
    for e in catalog:
        names = {normalize(v) for v in [e["name"]] + (e["variants"] or []) if v}
        matched = (names & requested) if requested else {n for n in names if n and f" {n} " in key_q}
        if matched:
            cands.append(e)
    if not requested:   # évite Google si seul « Google China » est cité
        cands = [e for e in cands if not any(
            normalize(e["name"]) != normalize(o["name"])
            and f" {normalize(e['name'])} " in f" {normalize(o['name'])} " for o in cands)]
    if not cands:
        raise HTTPException(422, "Aucune entité du graphe n'est citée dans la question. GraphRAG part d'une entité précise : reformulez avec son nom exact (ex. « Google », « Thomas Schäfer »). Les questions globales (classements, comptages) se font en Cypher.")
    if len(cands) > 5:
        raise HTTPException(422, "Trop d'entités citées dans la question (max 5). Réduisez-la à quelques entités.")
    seeds, seed_ids = cands, [e["entity_id"] for e in cands]

    edge_query = """
        MATCH (n:Entity)-[r]-(:Entity)
        WHERE n.entity_id IN $ids AND r.dataset_id = $dataset
          AND NOT r.relation_id IN $excluded
        WITH DISTINCT r ORDER BY r.relation_id LIMIT $limit""" + EDGE_RETURN
    first = cypher(edge_query, ids=seed_ids, dataset=DATASET, excluded=[], limit=q.max_relations + 1)
    truncated = len(first) > q.max_relations
    edges = first[:q.max_relations]
    if q.hops == 2 and not truncated:
        nb = sorted({x for e in edges for x in (e["source_id"], e["target_id"]) if x not in seed_ids})
        rem = q.max_relations - len(edges)
        if nb:
            second = cypher(edge_query, ids=nb, dataset=DATASET,
                            excluded=[e["relation_id"] for e in edges], limit=rem + 1)
            truncated = len(second) > rem
            edges.extend(second[:rem])
    retrieval_ms = round((perf_counter() - start) * 1000, 2)
    if not edges:
        raise HTTPException(404, "Aucune relation récupérée pour ces entités.")

    facts = []
    for i, edge in enumerate(edges, 1):
        recs = json.loads(edge.pop("evidence_records_json") or "[]")
        facts.append({"citation_id": f"S{i}", **edge, "sources": [{
            "quote": r["evidence"][:350], "source_url": r["provenance"].get("source_url"),
            "revision_url": r["provenance"].get("revision_url"),
            "revision_id": r["provenance"].get("revision_id"),
            "chunk_id": r["provenance"].get("chunk_id")} for r in recs[:2]]})

    # Budget : sur le palier gratuit Groq, entrée + sortie doivent tenir dans 8000 tokens
    # par requête (TPM). On réduit donc le contexte envoyé au modèle (estimation : 1 token ≈ 3 caractères).
    MAX_CHARS_TOTAL = 15000

    def llm_fact(f, qlen):
        quote = (f["sources"][0]["quote"] if f["sources"] else "")[:qlen]
        return {"citation_id": f["citation_id"], "subject": f["subject"], "relation": f["relation"],
                "object": f["object"], "role_title": f["role_title"], "start_date": f["start_date"],
                "end_date": f["end_date"], "temporal_status": f["temporal_status"], "quote": quote}

    def build(n, qlen):
        req = {"question": q.question, "context": {
            "truncated": truncated or n < len(facts), "seeds": [x["name"] for x in seeds],
            "facts": [llm_fact(f, qlen) for f in facts[:n]]}}
        return req, json.dumps(req, ensure_ascii=False, separators=(",", ":"))

    n, qlen = len(facts), 250
    request, payload = build(n, qlen)
    while len(SYSTEM_PROMPT) + len(payload) > MAX_CHARS_TOTAL and n > 3:
        if qlen > 80:
            qlen = max(80, qlen - 70)      # d'abord raccourcir les extraits
        else:
            n -= 1                          # puis retirer des relations
        request, payload = build(n, qlen)
    if n < len(facts):
        truncated, facts = True, facts[:n]

    settings = {"model": MODEL, "system_prompt": SYSTEM_PROMPT, "temperature": 0,
                "max_completion_tokens": 2000, "reasoning_effort": "low"}
    cache_key = hashlib.sha256(json.dumps({"settings": settings, "request": request},
                               ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    folder = CACHE / cache_key
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "groq_response.json"
    cached, gen_ms = raw_path.exists(), None

    if cached:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise HTTPException(500, "Configure GROQ_API_KEY dans .env.")
        t = perf_counter()
        try:
            resp = Groq(api_key=key, timeout=120, max_retries=0).chat.completions.create(
                model=MODEL, temperature=0, reasoning_effort="low", max_completion_tokens=2000,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": payload}])
        except RateLimitError as exc:
            raise HTTPException(429, f"Limite Groq atteinte. Retry-After : {exc.response.headers.get('retry-after')}")
        except APIStatusError as exc:
            raise HTTPException(502, f"Groq a refusé la requête (HTTP {exc.status_code}) : {str(exc.message)[:400]}")
        except APIError as exc:
            raise HTTPException(502, f"Groq injoignable ou erreur réseau : {str(exc)[:300]}")
        gen_ms = round((perf_counter() - t) * 1000, 2)
        raw = resp.model_dump(mode="json")
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop":
        raise HTTPException(502, f"Réponse incomplète : {choice['finish_reason']}")
    answer = choice["message"]["content"] or ""
    cited = sorted(set(re.findall(r"\[(S\d+)\]", answer)))
    known = {f["citation_id"] for f in facts}
    usage = raw.get("usage") or {}
    return {"answer": answer, "facts": facts, "seeds": [s["name"] for s in seeds],
            "cited": cited, "unknown_citations": sorted(set(cited) - known),
            "truncated": truncated, "cached": cached, "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms, "total_tokens": usage.get("total_tokens"),
            "latency_ms": round(retrieval_ms + (gen_ms or 0), 2), "model": MODEL}


@app.post("/api/graphrag")
def graphrag(q: RagIn):
    return run_graphrag(q)


# ---------- Benchmark ----------
def recall(expected, text):
    t = normalize(text)
    hits = [e for e in expected if normalize(e) in t]
    return {"score": len(hits) / len(expected) if expected else None, "hits": hits,
            "missing": [e for e in expected if e not in hits]}


@app.get("/api/benchmark")
def benchmark_items():
    return json.loads(BENCH.read_text(encoding="utf-8")) if BENCH.exists() else []


@app.post("/api/benchmark/run")
def benchmark_run():
    results = []
    for item in benchmark_items():
        row = {"id": item["id"], "question": item["question"], "type": item.get("type"),
               "expected": item["expected"]}
        try:
            a = item["analytics"]   # {"cypher": "..."} ou {"query_id": "..."} (+ "params")
            cy = a.get("cypher") or fill(QUERIES[a["query_id"]]["cypher"])
            res = run_cypher(cy)
            row["analytics"] = {**recall(item["expected"], json.dumps(res["rows"], ensure_ascii=False)),
                                "latency_ms": res["latency_ms"], "rows": res["row_count"]}
        except Exception as exc:
            row["analytics"] = {"error": str(getattr(exc, "detail", exc))}
        try:
            res = run_graphrag(RagIn(question=item["question"], entities=item.get("entities", [])))
            row["graphrag"] = {**recall(item["expected"], res["answer"]),
                               "latency_ms": res["latency_ms"], "cached": res["cached"],
                               "citations": len(res["cited"]), "unknown": len(res["unknown_citations"]),
                               "tokens": res["total_tokens"], "truncated": res["truncated"]}
            if not res["cached"]:
                time.sleep(25)  # ménage la limite de 8000 tokens/minute de Groq
        except Exception as exc:
            row["graphrag"] = {"error": str(getattr(exc, "detail", exc))}
        results.append(row)
    return results


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")