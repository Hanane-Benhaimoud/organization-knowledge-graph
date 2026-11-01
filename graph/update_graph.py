import argparse
import fcntl
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def run(*args, env=None):
    subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
    )


def wait_for_spark(timeout=120):
    print("Attente des deux workers Spark…", flush=True)

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            import json

            with urlopen(
                "http://127.0.0.1:8081/json/",
                timeout=3,
            ) as response:
                state = json.load(response)

            alive = [
                worker
                for worker in state.get("workers", [])
                if worker.get("state") == "ALIVE"
            ]

            if len(alive) >= 2:
                print("Deux workers Spark disponibles.", flush=True)
                return

        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError(
        "Les deux workers Spark ne sont pas disponibles. "
        "Vérifie Docker et http://localhost:8081."
    )


def wait_for_neo4j(timeout=120):
    password = os.getenv("NEO4J_PASSWORD", "").strip()

    if not password:
        raise ValueError("Configure NEO4J_PASSWORD dans .env.")

    print("Attente de Neo4j…", flush=True)

    deadline = time.monotonic() + timeout

    with GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", password),
        connection_timeout=3,
        connection_acquisition_timeout=5,
    ) as driver:
        while time.monotonic() < deadline:
            try:
                driver.verify_connectivity()
                print("Neo4j disponible.", flush=True)
                return

            except Exception:
                time.sleep(2)

    raise RuntimeError(
        "Neo4j reste indisponible. Vérifie son démarrage "
        "et le mot de passe existant."
    )


def update(import_neo4j):
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    gold_path = (
        f"/data/orggraph/gold/"
        f"mvp_{timestamp}_{uuid.uuid4().hex[:8]}"
    )

    print(f"\nDestination Gold : {gold_path}", flush=True)

    # Le NameNode doit déjà être initialisé et HDFS opérationnel.
    run(
        "docker", "compose", "up", "-d",
        "namenode", "datanode1", "datanode2",
    )

    print("\n1/3 — Synchronisation HDFS", flush=True)

    run(
        sys.executable,
        "-m",
        "extraction.upload_llm_hdfs",
    )

    print("\n2/3 — Préparation Spark", flush=True)

    run(
        "docker", "compose",
        "--profile", "compute",
        "up", "-d",
        "spark-master",
        "spark-worker1",
        "spark-worker2",
    )

    wait_for_spark()

    run(
        "docker", "compose",
        "--profile", "compute",
        "--profile", "jobs",
        "run",
        "--rm",
        "--no-deps",
        "--use-aliases",
        "spark-client",
        "/opt/orggraph/spark/prepare_graph.py",
        "--output",
        f"hdfs://namenode:8020{gold_path}",
    )

    if import_neo4j:
        print("\n3/3 — Synchronisation Neo4j", flush=True)

        run(
            "docker", "compose",
            "--profile", "graph",
            "up", "-d", "neo4j",
        )

        wait_for_neo4j()

        environment = os.environ.copy()
        environment["ORGGRAPH_GOLD_PATH"] = gold_path

        run(
            sys.executable,
            "-m",
            "graph.import_neo4j",
            env=environment,
        )

    print("\nMISE À JOUR TERMINÉE", flush=True)
    print(f"Gold : {gold_path}", flush=True)

    if import_neo4j:
        print(
            "Neo4j synchronisé. Recharge la page de l'interface "
            "et relance les analyses ou les questions souhaitées.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--import-neo4j",
        action="store_true",
        help="Synchroniser Neo4j après Spark.",
    )

    args = parser.parse_args()

    lock_dir = ROOT / "data" / "pipeline"
    lock_dir.mkdir(parents=True, exist_ok=True)

    with (lock_dir / "update_graph.lock").open("a") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit(
                "Une mise à jour du graphe est déjà en cours."
            )

        update(args.import_neo4j)


if __name__ == "__main__":
    main()