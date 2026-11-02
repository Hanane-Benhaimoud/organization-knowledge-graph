import argparse
import json
import uuid
from datetime import datetime, timezone

from pyspark.sql import SparkSession, functions as F, types as T


BASE = "hdfs://namenode:8020/data/orggraph"


def name_key(column):
    """Normalise les espaces et la casse, sans retirer les accents."""
    return F.lower(
        F.trim(
            F.regexp_replace(
                F.regexp_replace(column, "\u00a0", " "),
                r"\s+",
                " ",
            )
        )
    )


def stable_id(prefix, *columns):
    """Identifiant déterministe avec prise en compte des valeurs nulles."""
    values = [column.cast("string") for column in columns]

    return F.concat(
        F.lit(prefix),
        F.sha2(F.to_json(F.array(*values)), 256),
    )


def null_if_blank(column):
    value = F.trim(F.regexp_replace(column, r"\s+", " "))
    return F.when(F.length(value) > 0, value)


def require_unique(dataframe, column, label):
    missing = dataframe.filter(
        F.col(column).isNull()
        | (F.length(F.col(column)) == 0)
    ).limit(1).count()

    duplicated = (
        dataframe.groupBy(column)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )

    if missing or duplicated:
        raise RuntimeError(
            f"{label} : identifiants absents ou dupliqués."
        )


def require_links(left, left_column, right, right_column, label):
    targets = right.select(
        F.col(right_column).alias("_target")
    ).distinct()

    missing = (
        left.join(
            targets,
            left[left_column] == targets["_target"],
            "left_anti",
        )
        .limit(1)
        .count()
    )

    if missing:
        raise RuntimeError(f"Lien invalide : {label}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validated",
        required=True,
        help="Dossier HDFS produit par validate_extractions.py",
    )

    args = parser.parse_args()
    validated_path = args.validated.rstrip("/")

    version = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    output = f"{BASE}/gold/document_graph/{version}"

    spark = (
        SparkSession.builder
        .appName("OrgGraph-Build-Document-Graph")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .config("spark.hadoop.dfs.replication", "2")
        .config("spark.sql.jsonGenerator.ignoreNullFields", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # ------------------------------------------------------------
        # 1. Entrées validées
        # ------------------------------------------------------------
        report_lines = (
            spark.read.text(f"{validated_path}/report/part-*")
            .collect()
        )

        if len(report_lines) != 1:
            raise RuntimeError("Un rapport de validation unique est attendu.")

        validation_report = json.loads(report_lines[0]["value"])

        extractions = (
            spark.read.parquet(
                f"{validated_path}/parquet/valid_extractions"
            )
            .cache()
        )

        extraction_count = extractions.count()

        if extraction_count != validation_report["valid_extractions"]:
            raise RuntimeError(
                "Le nombre d'extractions diffère du rapport de validation."
            )

        if extraction_count == 0:
            raise RuntimeError("Aucune extraction valide.")

        # ------------------------------------------------------------
        # 2. Tous les segments : y compris ceux sans relation
        # ------------------------------------------------------------
        source_chunks = (
            spark.read
            .option("mode", "FAILFAST")
            .json(
                f"{BASE}/silver/preparation/wikipedia_chunks.jsonl"
            )
        )

        chunks = (
            source_chunks.select(
                "chunk_id",
                "document_id",
                F.col("wikidata_id").alias("article_qid"),
                F.col("entity_type").alias("article_entity_type"),
                F.col("entity_label").alias("article_entity_label"),
                "title",
                "language",
                "source",
                "source_url",
                "revision_url",
                F.col("revision_id").cast("long").alias("revision_id"),
                "collected_at",
                F.col("section_index").cast("long").alias("section_index"),
                "section_heading",
                F.col("part_index").cast("long").alias("part_index"),
                F.col("char_start").cast("long").alias("char_start"),
                F.col("char_end").cast("long").alias("char_end"),
                "text",
            )
            .withColumn(
                "document_version_id",
                stable_id(
                    "doc:",
                    F.col("document_id"),
                    F.col("revision_id"),
                ),
            )
            .withColumn("text_hash", F.sha2("text", 256))
            .withColumn(
                "embedding_text",
                F.concat_ws(
                    "\n",
                    F.col("title"),
                    F.col("section_heading"),
                    F.col("text"),
                ),
            )
            .cache()
        )

        require_unique(chunks, "chunk_id", "Segments")

        if chunks.filter(
            F.col("text").isNull()
            | (F.length(F.trim(F.col("text"))) == 0)
            | F.col("revision_id").isNull()
            | F.col("document_id").isNull()
        ).limit(1).count():
            raise RuntimeError(
                "Segment vide ou métadonnées documentaires manquantes."
            )

        # Les champs documentaires doivent être cohérents
        # entre les segments d'une même version d'article.
        document_fields = [
            "document_id",
            "article_qid",
            "title",
            "language",
            "source",
            "source_url",
            "revision_url",
            "revision_id",
        ]

        conflicts = (
            chunks.groupBy("document_version_id")
            .agg(
                F.countDistinct(
                    F.to_json(F.struct(*document_fields))
                ).alias("versions")
            )
            .filter(F.col("versions") > 1)
            .limit(1)
            .count()
        )

        if conflicts:
            raise RuntimeError(
                "Métadonnées contradictoires pour une version documentaire."
            )

        documents = (
            chunks.groupBy(
                "document_version_id",
                *document_fields,
            )
            .agg(
                F.min("collected_at").alias("collected_at"),
                F.count("*").alias("chunk_count"),
            )
            .cache()
        )

        # ------------------------------------------------------------
        # 3. Référentiel Wikidata
        # ------------------------------------------------------------
        reference_schema = T.StructType([
            T.StructField(
                "entities",
                T.ArrayType(
                    T.StructType([
                        T.StructField("qid", T.StringType()),
                        T.StructField("label", T.StringType()),
                        T.StructField("entity_type", T.StringType()),
                        T.StructField("description", T.StringType()),
                    ])
                ),
            )
        ])

        reference = (
            spark.read
            .schema(reference_schema)
            .option("multiLine", True)
            .option("mode", "FAILFAST")
            .json(f"{BASE}/bronze/wikidata/entities.json")
            .select(F.explode("entities").alias("e"))
            .select("e.*")
            .cache()
        )

        require_unique(reference, "qid", "Référentiel Wikidata")

        reference_typed = reference.filter(
            F.col("entity_type").isin("person", "organization")
            & F.col("qid").rlike(r"^Q[1-9][0-9]*$")
            & (F.length(F.trim(F.col("label"))) > 0)
        )

        # Les noms disponibles dans le référentiel.
        label_aliases = reference_typed.select(
            "qid",
            "entity_type",
            F.col("label").alias("alias"),
        )

        # Titres et labels des articles dont le QID et le type
        # sont présents dans le référentiel.
        article_aliases = (
            chunks.select(
                F.col("article_qid").alias("qid"),
                F.explode(
                    F.array("title", "article_entity_label")
                ).alias("alias"),
            )
            .join(
                reference_typed.select("qid", "entity_type"),
                "qid",
                "inner",
            )
            .select("qid", "entity_type", "alias")
        )

        aliases = (
            label_aliases.unionByName(article_aliases)
            .withColumn("name_key", name_key(F.col("alias")))
            .filter(
                F.col("name_key").isNotNull()
                & (F.length("name_key") > 0)
            )
            .select("qid", "entity_type", "name_key")
            .distinct()
        )

        candidates = (
            aliases.groupBy("entity_type", "name_key")
            .agg(
                F.sort_array(
                    F.collect_set("qid")
                ).alias("candidate_qids")
            )
            .withColumn(
                "candidate_count",
                F.size("candidate_qids"),
            )
        )

        # ------------------------------------------------------------
        # 4. Rattachement aux segments courants
        # ------------------------------------------------------------
        flat = extractions.select(
            F.col("relation.subject").alias("subject"),
            F.col("relation.subject_type").alias("subject_type"),
            F.col("relation.relation").alias("relation_type"),
            F.col("relation.object").alias("object"),
            F.col("relation.object_type").alias("object_type"),
            F.col("relation.role_title").alias("role_title"),
            F.col("relation.start_date").alias("start_date"),
            F.col("relation.end_date").alias("end_date"),
            F.col("relation.temporal_status").alias("temporal_status"),
            F.col("relation.evidence").alias("quote"),
            F.col("provenance"),
            F.col("provenance.chunk_id").alias("chunk_id"),
            F.col("source_text").alias("_validated_text"),
        )

        linked = flat.join(
            chunks.select(
                "chunk_id",
                "document_id",
                "document_version_id",
                "revision_id",
                "source_url",
                "revision_url",
                "text",
            ),
            "chunk_id",
            "left",
        ).cache()

        if linked.filter(
            F.col("document_version_id").isNull()
            | ~F.col("_validated_text").eqNullSafe(F.col("text"))
            | ~F.col("provenance.document_id").eqNullSafe(
                F.col("document_id")
            )
            | ~F.col("provenance.revision_id").eqNullSafe(
                F.col("revision_id")
            )
        ).limit(1).count():
            raise RuntimeError(
                "Les segments ne correspondent plus à la validation. "
                "Relance la validation avant de continuer."
            )

        # ------------------------------------------------------------
        # 5. Résolution des extrémités
        # ------------------------------------------------------------
        subjects = linked.select(
            "document_version_id",
            F.col("subject_type").alias("entity_type"),
            F.col("subject").alias("surface"),
        )

        objects = linked.select(
            "document_version_id",
            F.col("object_type").alias("entity_type"),
            F.col("object").alias("surface"),
        )

        endpoint_names = (
            subjects.unionByName(objects)
            .withColumn("name_key", name_key(F.col("surface")))
            .groupBy(
                "document_version_id",
                "entity_type",
                "name_key",
            )
            .agg(
                F.min("surface").alias("display_name"),
                F.sort_array(
                    F.collect_set("surface")
                ).alias("observed_names"),
            )
        )

        resolution = (
            endpoint_names.join(
                candidates,
                ["entity_type", "name_key"],
                "left",
            )
            .withColumn(
                "candidate_count",
                F.coalesce("candidate_count", F.lit(0)),
            )
            .withColumn(
                "candidate_qids",
                F.coalesce(
                    "candidate_qids",
                    F.array().cast("array<string>"),
                ),
            )
            .withColumn(
                "qid",
                F.when(
                    F.col("candidate_count") == 1,
                    F.element_at("candidate_qids", 1),
                ),
            )
            .withColumn(
                "resolution_method",
                F.when(
                    F.col("candidate_count") == 1,
                    F.lit("exact_unique_reference_match"),
                )
                .when(
                    F.col("candidate_count") > 1,
                    F.lit("ambiguous_document_scoped"),
                )
                .otherwise(F.lit("unmatched_document_scoped")),
            )
            .withColumn(
                "entity_id",
                F.when(
                    F.col("qid").isNotNull(),
                    F.concat(F.lit("wikidata:"), F.col("qid")),
                ).otherwise(
                    stable_id(
                        "local:",
                        F.col("document_version_id"),
                        F.col("entity_type"),
                        F.col("name_key"),
                    )
                ),
            )
            .cache()
        )

        # Les entités sans QID sont regroupées uniquement à
        # l'intérieur d'un même document et d'un même type.
        grouped_entities = (
            resolution.groupBy(
                "entity_id",
                "entity_type",
                "qid",
                "resolution_method",
            )
            .agg(
                F.min("display_name").alias("_display_name"),
                F.sort_array(
                    F.array_distinct(
                        F.flatten(F.collect_list("observed_names"))
                    )
                ).alias("name_variants"),
                F.sort_array(
                    F.collect_set("document_version_id")
                ).alias("source_document_version_ids"),
            )
        )

        entities = (
            grouped_entities.join(
                reference_typed.select(
                    "qid",
                    F.col("label").alias("_reference_label"),
                    F.col("description").alias("description"),
                ),
                "qid",
                "left",
            )
            .withColumn(
                "name",
                F.coalesce("_reference_label", "_display_name"),
            )
            .withColumn("name_key", name_key(F.col("name")))
            .withColumn(
                "identity_manually_verified",
                F.lit(False),
            )
            .drop("_display_name", "_reference_label")
            .cache()
        )

        # Rattache les identifiants résolus à chaque extraction.
        resolved = linked

        for side in ("subject", "object"):
            mapping = resolution.select(
                F.col("document_version_id").alias(f"_{side}_doc"),
                F.col("entity_type").alias(f"_{side}_type"),
                F.col("name_key").alias(f"_{side}_key"),
                F.col("entity_id").alias(f"{side}_id"),
            )

            resolved = (
                resolved.join(
                    mapping,
                    (
                        resolved["document_version_id"]
                        == mapping[f"_{side}_doc"]
                    )
                    & (
                        resolved[f"{side}_type"]
                        == mapping[f"_{side}_type"]
                    )
                    & (
                        name_key(resolved[side])
                        == mapping[f"_{side}_key"]
                    ),
                    "left",
                )
                .drop(
                    f"_{side}_doc",
                    f"_{side}_type",
                    f"_{side}_key",
                )
            )

        resolved = resolved.cache()

        if resolved.count() != extraction_count:
            raise RuntimeError(
                "La résolution a modifié le nombre d'extractions."
            )

        # ------------------------------------------------------------
        # 6. Mentions dérivées des extrémités extraites
        # ------------------------------------------------------------
        mention_frames = []

        for side in ("subject", "object"):
            mention_frames.append(
                resolved.select(
                    "chunk_id",
                    "document_version_id",
                    F.col(f"{side}_id").alias("entity_id"),
                    F.col(side).alias("surface"),
                )
            )

        mentions = (
            mention_frames[0]
            .unionByName(mention_frames[1])
            .groupBy(
                "chunk_id",
                "document_version_id",
                "entity_id",
            )
            .agg(
                F.sort_array(
                    F.collect_set("surface")
                ).alias("surface_forms")
            )
            .withColumn(
                "mention_id",
                stable_id(
                    "mention:",
                    F.col("chunk_id"),
                    F.col("entity_id"),
                ),
            )
            .withColumn(
                "mention_method",
                F.lit("validated_relation_endpoint"),
            )
            .cache()
        )

        # ------------------------------------------------------------
        # 7. Relations et preuves
        # ------------------------------------------------------------
        enriched = (
            resolved
            .withColumn("role_title", null_if_blank(F.col("role_title")))
            .withColumn("start_date", null_if_blank(F.col("start_date")))
            .withColumn("end_date", null_if_blank(F.col("end_date")))
        )

        identity = [
            "subject_id",
            "relation_type",
            "object_id",
            "role_title",
            "start_date",
            "end_date",
            "temporal_status",
        ]

        enriched = (
            enriched.withColumn(
                "relation_id",
                stable_id(
                    "relation:",
                    *[F.col(name) for name in identity],
                ),
            )
            .cache()
        )

        relations = (
            enriched.groupBy("relation_id", *identity)
            .agg(
                F.count("*").alias("extraction_count"),
                F.countDistinct("chunk_id").alias(
                    "supporting_chunk_count"
                ),
                F.countDistinct("document_version_id").alias(
                    "supporting_document_count"
                ),
            )
            .withColumn(
                "self_relation",
                F.col("subject_id") == F.col("object_id"),
            )
            .cache()
        )

        # La position est relative au texte du segment.
        evidence = (
            enriched.select(
                "relation_id",
                "chunk_id",
                "document_version_id",
                "quote",
                "source_url",
                "revision_url",
                "revision_id",
                F.col("provenance.run_id").alias("extraction_run_id"),
                F.col("provenance.model").alias("extraction_model"),
                (
                    F.expr("instr(text, quote)") - F.lit(1)
                ).alias("quote_start_in_chunk"),
            )
            .withColumn(
                "quote_end_in_chunk",
                F.col("quote_start_in_chunk") + F.length("quote"),
            )
            .withColumn(
                "evidence_id",
                stable_id(
                    "evidence:",
                    F.col("relation_id"),
                    F.col("chunk_id"),
                    F.col("quote"),
                ),
            )
            .dropDuplicates(["evidence_id"])
            .cache()
        )

        # ------------------------------------------------------------
        # 8. Contrôles d'intégrité
        # ------------------------------------------------------------
        tables = {
            "documents": (documents, "document_version_id"),
            "chunks": (chunks, "chunk_id"),
            "entities": (entities, "entity_id"),
            "mentions": (mentions, "mention_id"),
            "relations": (relations, "relation_id"),
            "evidence": (evidence, "evidence_id"),
        }

        counts = {}

        for name, (dataframe, identifier) in tables.items():
            require_unique(dataframe, identifier, name)
            counts[name] = dataframe.count()

        for side in ("subject_id", "object_id"):
            require_links(
                relations, side,
                entities, "entity_id",
                f"relations.{side} -> entities",
            )

        require_links(
            chunks, "document_version_id",
            documents, "document_version_id",
            "chunks -> documents",
        )

        require_links(
            mentions, "chunk_id",
            chunks, "chunk_id",
            "mentions -> chunks",
        )

        require_links(
            mentions, "entity_id",
            entities, "entity_id",
            "mentions -> entities",
        )

        require_links(
            evidence, "relation_id",
            relations, "relation_id",
            "evidence -> relations",
        )

        require_links(
            evidence, "chunk_id",
            chunks, "chunk_id",
            "evidence -> chunks",
        )

        if evidence.filter(
            F.col("quote_start_in_chunk") < 0
        ).limit(1).count():
            raise RuntimeError("Une preuve est absente du texte source.")

        # ------------------------------------------------------------
        # 9. Écriture Gold, sans modification de Neo4j
        # ------------------------------------------------------------
        for name, (dataframe, _) in tables.items():
            dataframe.write.mode("errorifexists").parquet(
                f"{output}/parquet/{name}"
            )

            dataframe.coalesce(1).write.mode(
                "errorifexists"
            ).json(f"{output}/json/{name}")

        resolution.coalesce(1).write.mode(
            "errorifexists"
        ).json(f"{output}/audit/entity_resolution")

        aliases.coalesce(1).write.mode(
            "errorifexists"
        ).json(f"{output}/audit/reference_aliases")

        resolution_summary = [
            row.asDict()
            for row in (
                entities.groupBy("resolution_method")
                .count()
                .orderBy("resolution_method")
                .collect()
            )
        ]

        report = {
            "validation_input": validated_path,
            "llm_run_id": validation_report["run_id"],
            "input_valid_extractions": extraction_count,
            "tables": counts,
            "reference_entities": reference.count(),
            "resolution_summary": resolution_summary,
            "self_relations_to_review": relations.filter(
                F.col("self_relation")
            ).count(),
            "all_integrity_checks_passed": True,
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

        print("\n===== RAPPORT GRAPHE DOCUMENTAIRE =====")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()