import argparse
import json
import uuid
from datetime import datetime, timezone

from pyspark.sql import SparkSession, functions as F, types as T


BASE = "hdfs://namenode:8020/data/orggraph"
DEFAULT_RUN_ID = "ff7e5205b7ffa21f"


def text_fields(names):
    return [
        T.StructField(name, T.StringType(), True)
        for name in names
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = (
        f"{BASE}/silver/validated/{args.run_id}/"
        f"{timestamp}_{uuid.uuid4().hex[:8]}"
    )

    relation_schema = T.StructType(text_fields([
        "subject",
        "subject_type",
        "relation",
        "object",
        "object_type",
        "role_title",
        "start_date",
        "end_date",
        "temporal_status",
        "evidence",
    ]))

    provenance_schema = T.StructType(
        text_fields([
            "chunk_id",
            "document_id",
            "wikidata_id",
            "entity_label",
            "entity_type",
            "source",
            "source_url",
            "revision_url",
            "language",
            "title",
            "collected_at",
            "section_heading",
            "run_id",
            "model",
            "prompt_version",
        ])
        + [
            T.StructField(
                "seed_categories",
                T.ArrayType(T.StringType()),
                True,
            )
        ]
        + [
            T.StructField(name, T.LongType(), True)
            for name in [
                "revision_id",
                "section_index",
                "part_index",
                "char_start",
                "char_end",
                "char_count",
            ]
        ]
    )

    extraction_schema = T.StructType([
        T.StructField("relation", relation_schema, True),
        T.StructField("provenance", provenance_schema, True),
    ])

    chunk_schema = T.StructType(
        text_fields(["chunk_id", "document_id", "text"])
        + [
            T.StructField("revision_id", T.LongType(), True)
        ]
    )

    spark = (
        SparkSession.builder
        .appName("OrgGraph-Validate-Extractions")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .config("spark.hadoop.dfs.replication", "2")
        .config("spark.sql.jsonGenerator.ignoreNullFields", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        llm_path = f"{BASE}/silver/llm/{args.run_id}"

        report_rows = (
            spark.read
            .option("multiLine", True)
            .json(f"{llm_path}/extraction_report.json")
            .collect()
        )

        if len(report_rows) != 1:
            raise RuntimeError("Un rapport LLM unique est attendu.")

        llm_report = report_rows[0].asDict(recursive=True)

        if llm_report.get("run_id") != args.run_id:
            raise RuntimeError("Le run_id du rapport est incohérent.")

        if llm_report.get("aggregate_files_updated") is not True:
            raise RuntimeError(
                "Les fichiers récapitulatifs LLM ne sont pas à jour."
            )

        raw = (
            spark.read
            .schema(extraction_schema)
            .option("mode", "FAILFAST")
            .json(f"{llm_path}/relations.jsonl")
            .cache()
        )

        chunks = (
            spark.read
            .schema(chunk_schema)
            .option("mode", "FAILFAST")
            .json(
                f"{BASE}/silver/preparation/"
                "wikipedia_chunks.jsonl"
            )
            .cache()
        )

        input_count = raw.count()

        if input_count != llm_report["accepted_relations"]:
            raise RuntimeError(
                "Le nombre de relations diffère du rapport LLM."
            )

        if input_count == 0:
            raise RuntimeError("Aucune extraction disponible.")

        missing_chunk_ids = chunks.filter(
            F.col("chunk_id").isNull()
            | (F.length(F.trim(F.col("chunk_id"))) == 0)
        ).count()

        duplicate_chunk_ids = (
            chunks.groupBy("chunk_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        if missing_chunk_ids or duplicate_chunk_ids:
            raise RuntimeError(
                "Identifiants de segments absents ou dupliqués."
            )

        joined = (
            raw.alias("e")
            .join(
                chunks.alias("c"),
                F.col("e.provenance.chunk_id")
                == F.col("c.chunk_id"),
                "left",
            )
            .select(
                F.col("e.relation"),
                F.col("e.provenance"),
                F.col("c.chunk_id").alias("source_chunk_id"),
                F.col("c.document_id").alias("source_document_id"),
                F.col("c.revision_id").alias("source_revision_id"),
                F.col("c.text").alias("source_text"),
            )
        )

        kind = F.col("relation.relation")
        subject_type = F.col("relation.subject_type")
        object_type = F.col("relation.object_type")

        allowed_types = (
            (
                kind.isin(
                    "CEO_OF",
                    "CHAIRPERSON_OF",
                    "BOARD_MEMBER_OF",
                    "EMPLOYED_BY",
                )
                & (subject_type == "person")
                & (object_type == "organization")
            )
            | (
                kind.isin("PARENT_OF", "PUBLISHED_BY")
                & (subject_type == "organization")
                & (object_type == "organization")
            )
            | (
                kind.isin("FOUNDED", "MEMBER_OF")
                & subject_type.isin("person", "organization")
                & (object_type == "organization")
            )
            | (
                (kind == "OWNED_BY")
                & (subject_type == "organization")
                & object_type.isin("person", "organization")
            )
        )

        errors = []

        def check(condition, reason):
            errors.append(
                F.when(
                    ~F.coalesce(condition, F.lit(False)),
                    F.lit(reason),
                )
            )

        check(
            F.length(F.trim(F.col("relation.subject"))) > 0,
            "empty_subject",
        )

        check(
            F.length(F.trim(F.col("relation.object"))) > 0,
            "empty_object",
        )

        check(
            allowed_types,
            "invalid_relation_or_endpoint_types",
        )

        check(
            F.col("source_chunk_id").isNotNull(),
            "missing_source_chunk",
        )

        check(
            F.col("provenance.document_id")
            == F.col("source_document_id"),
            "document_mismatch",
        )

        check(
            F.col("provenance.revision_id")
            == F.col("source_revision_id"),
            "revision_mismatch",
        )

        check(
            F.col("provenance.run_id") == args.run_id,
            "run_id_mismatch",
        )

        check(
            F.length(F.trim(F.col("relation.evidence"))) > 0,
            "empty_evidence",
        )

        check(
            F.expr("instr(source_text, relation.evidence) > 0"),
            "evidence_not_in_source",
        )

        check(
            F.col("relation.temporal_status").isin(
                "current_at_source", "former", "unknown"
            ),
            "invalid_temporal_status",
        )

        check(
            F.col("provenance.source_url").rlike(r"^https?://"),
            "invalid_source_url",
        )

        # Contrôle du format, pas de la vérité historique
        # ni de toutes les règles du calendrier.
        date_pattern = (
            r"^\d{4}"
            r"(-(0[1-9]|1[0-2])"
            r"(-(0[1-9]|[12]\d|3[01]))?)?$"
        )

        for name in ("start_date", "end_date"):
            value = F.col(f"relation.{name}")

            check(
                value.isNull() | value.rlike(date_pattern),
                f"invalid_{name}_format",
            )

        audited = (
            joined.withColumn(
                "validation_errors",
                F.filter(
                    F.array(*errors),
                    lambda value: value.isNotNull(),
                ),
            )
            .cache()
        )

        valid = (
            audited.filter(F.size("validation_errors") == 0)
            .cache()
        )

        quarantine = (
            audited.filter(F.size("validation_errors") > 0)
            .cache()
        )

        valid_count = valid.count()
        quarantine_count = quarantine.count()

        if valid_count + quarantine_count != input_count:
            raise RuntimeError("Bilan de validation incohérent.")

        reasons = (
            quarantine
            .select(
                F.explode("validation_errors").alias("reason")
            )
            .groupBy("reason")
            .count()
            .orderBy(F.desc("count"))
        )

        for name, dataframe in (
            ("valid_extractions", valid),
            ("quarantine", quarantine),
        ):
            dataframe.write.mode("errorifexists").parquet(
                f"{output}/parquet/{name}"
            )

            dataframe.coalesce(1).write.mode(
                "errorifexists"
            ).json(f"{output}/json/{name}")

        report = {
            "run_id": args.run_id,
            "llm_chunks_validated": llm_report["chunks_validated"],
            "chunks_available": chunks.count(),
            "input_extractions": input_count,
            "valid_extractions": valid_count,
            "quarantined_extractions": quarantine_count,
            "supporting_chunks": valid.select(
                "source_chunk_id"
            ).distinct().count(),
            "supporting_documents": valid.select(
                "source_document_id"
            ).distinct().count(),
            "rejection_reasons": [
                row.asDict()
                for row in reasons.collect()
            ],
            "output": output,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

        (
            spark.createDataFrame(
                [(json.dumps(report, ensure_ascii=False),)],
                ["value"],
            )
            .coalesce(1)
            .write.mode("errorifexists")
            .text(f"{output}/report")
        )

        print("\n===== RAPPORT DE VALIDATION =====")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()