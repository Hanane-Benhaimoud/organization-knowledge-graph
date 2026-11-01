import html
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

from graph.analytics import QUERIES


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET = "mvp_ff7e5205b7ffa21f_1139"

st.set_page_config(
    page_title="OrgGraph | Knowledge Intelligence",
    page_icon="🔎",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #f6f8fc;
}
h1, h2, h3 {
    color: #14243b;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e8f0;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stSidebar"] {
    background: #eaf0f8;
}
.hero {
    padding: 26px;
    background: linear-gradient(115deg, #10243d, #174c75);
    color: white;
    border-radius: 18px;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    margin: 0;
}
.hero p {
    color: #d8e6f6;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_driver():
    password = os.getenv("NEO4J_PASSWORD", "").strip()

    if not password:
        raise RuntimeError("NEO4J_PASSWORD manque dans .env.")

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", password),
    )
    driver.verify_connectivity()
    return driver


def query(cypher, **parameters):
    records, _, _ = get_driver().execute_query(
        cypher,
        parameters_=parameters,
        database_="neo4j",
        routing_="r",
    )
    return [record.data() for record in records]


@st.cache_data(ttl=30)
def entity_catalog():
    return query("""
        MATCH (n:Entity)-[r]-(:Entity)
        WHERE r.dataset_id = $dataset
        RETURN DISTINCT n.entity_id AS id,
               n.name AS name,
               n.entity_type AS entity_type
        ORDER BY name, id
    """, dataset=DATASET)


def download_json(label, value, filename, key):
    st.download_button(
        label,
        data=json.dumps(value, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        key=key,
    )


def run_rag(question, entity, hops):
    # Réutilise le programme existant et son cache.
    # Aucun shell : les textes sont transmis comme arguments.
    command = [
        sys.executable,
        "-m", "graph.graphrag",
        question,
        "--entity", entity,
        "--hops", str(hops),
        "--max-relations", "60",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Délai dépassé. Tu peux réessayer ; une réponse "
            "déjà sauvegardée sera réutilisée."
        )

    prefix = "Résultat sauvegardé : "

    paths = [
        line[len(prefix):].strip()
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]

    if result.returncode != 0 or not paths:
        # Les erreurs attendues du script sont affichées sur stdout/stderr.
        message = result.stdout + "\n" + result.stderr

        for variable in ("GROQ_API_KEY", "NEO4J_PASSWORD"):
            secret = os.getenv(variable, "")
            if secret:
                message = message.replace(secret, "[REDACTED]")

        raise RuntimeError(message[-2500:] or "Échec de GraphRAG.")

    path = Path(paths[-1]).resolve()
    allowed = (ROOT / "data" / "graphrag").resolve()

    if not path.is_relative_to(allowed):
        raise RuntimeError("Chemin de résultat inattendu.")

    return json.loads(path.read_text(encoding="utf-8"))


def display_rag(result, key):
    st.markdown(result["answer"])

    columns = st.columns(3)
    columns[0].metric(
        "Relations dans le contexte",
        len(result["context"]["facts"]),
    )
    columns[1].metric(
        "Récupération",
        f"{result['retrieval_ms']:.1f} ms",
    )

    generation = result.get("generation_ms")
    columns[2].metric(
        "Génération",
        "Cache" if generation is None else f"{generation:.1f} ms",
    )

    if result.get("context_truncated"):
        st.warning(
            "Le contexte a été limité : cette réponse "
            "ne doit pas être considérée comme exhaustive."
        )

    if result.get("unknown_citation_ids"):
        st.warning("La réponse contient des identifiants de citation inconnus.")

    if not result.get("has_citations"):
        st.warning("La réponse ne contient aucune référence [Sx].")

    with st.expander("Consulter les relations et leurs preuves"):
        for fact in result["context"]["facts"]:
            st.markdown(
                f"**[{fact['citation_id']}] "
                f"{fact['subject']} → {fact['relation']} → "
                f"{fact['object']}**"
            )

            st.caption(
                f"Statut : {fact.get('temporal_status')} | "
                f"Début : {fact.get('start_date') or 'inconnu'} | "
                f"Fin : {fact.get('end_date') or 'inconnue'}"
            )

            for source in fact["sources"]:
                st.write(source["quote"])
                url = source.get("revision_url") or source.get("source_url")
                if url:
                    st.markdown(f"[Document source]({url})")

    download_json(
        "Télécharger la réponse et son contexte",
        result,
        "graphrag_result.json",
        key,
    )


st.sidebar.title("🔎 OrgGraph")
st.sidebar.caption("Knowledge Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d’ensemble",
        "Explorer le graphe",
        "Analyses Cypher",
        "Assistant GraphRAG",
        "Comparaison",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Version du graphe")
st.sidebar.code(DATASET, language=None)

if st.sidebar.button("Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Les fonctions « en cours » le sont selon leurs sources, "
    "pas nécessairement aujourd’hui."
)

try:
    get_driver()
    catalog = entity_catalog()
except Exception:
    st.error(
        "Connexion Neo4j impossible. Vérifie que le service est démarré "
        "et que NEO4J_PASSWORD correspond au mot de passe de la base."
    )
    st.stop()

if not catalog:
    st.warning("Aucune entité trouvée pour cette version du graphe.")
    st.stop()

by_id = {item["id"]: item for item in catalog}
people = [item for item in catalog if item["entity_type"] == "person"]
organizations = [
    item for item in catalog if item["entity_type"] == "organization"
]


def entity_label(entity_id):
    item = by_id[entity_id]
    return f"{item['name']} · {item['entity_type']}"


if page == "Vue d’ensemble":
    st.markdown("""
    <div class="hero">
        <h1>OrgGraph</h1>
        <p>Explorez les organisations, leurs dirigeants et les liens
        qui les relient à partir de documents publics.</p>
    </div>
    """, unsafe_allow_html=True)

    relation_counts = query("""
        MATCH (:Entity)-[r]->(:Entity)
        WHERE r.dataset_id = $dataset
        RETURN type(r) AS relation, count(r) AS count
        ORDER BY count DESC
    """, dataset=DATASET)

    columns = st.columns(4)
    columns[0].metric("Entités connectées", len(catalog))
    columns[1].metric("Personnes", len(people))
    columns[2].metric("Organisations", len(organizations))
    columns[3].metric(
        "Relations",
        sum(row["count"] for row in relation_counts),
    )

    st.subheader("Relations présentes dans le graphe")
    if relation_counts:
        frame = pd.DataFrame(relation_counts).set_index("relation")
        st.bar_chart(frame["count"])

    st.subheader("Deux façons d’interroger les mêmes données")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Analyses Cypher")
        st.write(
            "Des résultats structurés pour explorer les affiliations, "
            "les dirigeants partagés et les chaînes de filiales."
        )

    with right:
        st.markdown("#### Assistant GraphRAG")
        st.write(
            "Des réponses rédigées à partir du voisinage récupéré "
            "dans Neo4j, accompagnées des citations disponibles."
        )

    st.info(
        "Ce MVP utilise les extractions Wikipédia importées. "
        "Les alias, les homonymes et certaines classifications "
        "d’entités restent à améliorer."
    )


elif page == "Explorer le graphe":
    st.title("Explorer le graphe")

    selected = st.selectbox(
        "Rechercher une personne ou une organisation",
        list(by_id),
        format_func=entity_label,
    )

    limit = st.slider("Nombre maximal de relations", 10, 150, 50)

    rows = query("""
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE (a.entity_id = $id OR b.entity_id = $id)
          AND r.dataset_id = $dataset
        RETURN a.entity_id AS source_id,
               a.name AS source,
               a.entity_type AS source_type,
               b.entity_id AS target_id,
               b.name AS target,
               b.entity_type AS target_type,
               type(r) AS relation,
               r.temporal_status AS status,
               r.role_title AS role,
               r.evidence AS evidence,
               r.source_urls AS source_urls
        ORDER BY relation, source, target
        LIMIT $limit
    """, id=selected, dataset=DATASET, limit=limit)

    st.caption(
        "Bleu : organisation · Vert : personne · "
        "Déplace les nœuds et utilise la molette pour zoomer."
    )

    network = Network(
        height="580px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#14243b",
        cdn_resources="in_line",
    )

    added = set()

    for row in rows:
        for side in ("source", "target"):
            node_id = row[f"{side}_id"]

            if node_id not in added:
                node_type = row[f"{side}_type"]

                network.add_node(
                    node_id,
                    label=row[side],
                    title=html.escape(row[side]),
                    color=(
                        "#34b68b" if node_type == "person"
                        else "#448aff"
                    ),
                    size=30 if node_id == selected else 20,
                )
                added.add(node_id)

        network.add_edge(
            row["source_id"],
            row["target_id"],
            label=row["relation"],
            title=html.escape(
                f"{row['relation']} | {row['status']} | "
                f"{row['role'] or ''}"
            ),
        )

    network.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 150},
        },
        "edges": {
            "smooth": {"enabled": True, "type": "dynamic"},
            "font": {"size": 10},
        },
        "interaction": {"hover": True},
    }))

    if rows:
        components.html(network.generate_html(), height=600)
        st.caption(
            f"{len(rows)} relations affichées. "
            "L’affichage peut être limité par le seuil choisi."
        )

        with st.expander("Voir les données et les preuves"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Aucune relation disponible.")


elif page == "Analyses Cypher":
    st.title("Analyses du graphe")

    labels = {
        "person_affiliations": "Affiliations d’une personne",
        "multiple_affiliations": "Personnes aux affiliations multiples",
        "shared_leaders": "Organisations partageant des dirigeants",
        "connected_entities": "Entités les plus connectées",
        "subsidiary_chains": "Chaînes de filiales",
    }

    with st.form("analytics_form"):
        analysis = st.selectbox(
            "Analyse",
            list(QUERIES),
            format_func=lambda value: labels[value],
        )

        person = st.text_input("Personne", "Thomas Schäfer")
        organization = st.text_input("Organisation", "Google")
        submitted = st.form_submit_button("Exécuter l’analyse")

    if submitted:
        definition = QUERIES[analysis]
        started = perf_counter()

        rows = query(
            definition["cypher"],
            dataset=DATASET,
            person=person,
            organization=organization,
        )

        st.session_state["analytics_result"] = {
            "analysis": analysis,
            "question": definition["question"],
            "cypher": definition["cypher"],
            "parameters": {
                "dataset": DATASET,
                "person": person,
                "organization": organization,
            },
            "rows": rows,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    result = st.session_state.get("analytics_result")

    if result:
        st.write(result["question"])
        st.caption(
            f"{len(result['rows'])} lignes · "
            f"{result['latency_ms']} ms"
        )

        st.dataframe(
            pd.DataFrame(result["rows"]),
            use_container_width=True,
        )

        with st.expander("Voir la requête Cypher"):
            st.code(result["cypher"], language="cypher")

        download_json(
            "Télécharger les résultats",
            result,
            "analytics_result.json",
            "download_analytics",
        )

    st.caption(
        "Ces analyses incluent les relations historiques et les périodes "
        "inconnues. Un lien partagé ne garantit pas des mandats simultanés."
    )


elif page == "Assistant GraphRAG":
    st.title("Assistant GraphRAG")
    st.write("Pose une question sur une entité présente dans le graphe.")

    with st.form("rag_form"):
        selected = st.selectbox(
            "Entité de départ",
            list(by_id),
            format_func=entity_label,
        )

        question = st.text_area(
            "Question",
            "À quelles organisations cette personne est-elle liée ? "
            "Distingue les fonctions anciennes et celles présentées "
            "comme en cours dans les sources.",
        )

        hops = st.radio(
            "Profondeur de recherche",
            [1, 2],
            horizontal=True,
        )

        submitted = st.form_submit_button("Générer la réponse")

    st.caption(
        "Un saut : relations directes. Deux sauts : voisinage élargi. "
        "La première génération consomme des tokens Groq."
    )

    if submitted:
        if not question.strip():
            st.warning("Saisis une question.")
        else:
            try:
                with st.spinner("Recherche dans le graphe et génération…"):
                    result = run_rag(
                        question.strip(),
                        by_id[selected]["name"],
                        hops,
                    )

                st.session_state["rag_result"] = result
            except RuntimeError as exc:
                st.error(str(exc))

    if "rag_result" in st.session_state:
        result = st.session_state["rag_result"]
        st.subheader("Dernière réponse")
        st.caption(result["question"])
        display_rag(result, "download_rag")


elif page == "Comparaison":
    st.title("Cypher et GraphRAG")
    st.write(
        "Compare les affiliations structurées d’une personne "
        "avec une réponse rédigée à partir du graphe."
    )

    if not people:
        st.info("Aucune personne disponible.")
        st.stop()

    with st.form("comparison_form"):
        selected = st.selectbox(
            "Personne",
            [person["id"] for person in people],
            format_func=entity_label,
        )

        submitted = st.form_submit_button("Lancer la comparaison")

    if submitted:
        name = by_id[selected]["name"]

        question = (
            f"À quelles organisations {name} est-il ou est-elle lié(e) ? "
            "Indique les fonctions et les périodes connues. "
            "Distingue les fonctions anciennes et celles présentées "
            "comme en cours dans les sources."
        )

        try:
            with st.spinner("Exécution des deux approches…"):
                started = perf_counter()

                rows = query(
                    QUERIES["person_affiliations"]["cypher"],
                    dataset=DATASET,
                    person=name,
                )

                cypher_ms = round(
                    (perf_counter() - started) * 1000,
                    2,
                )

                rag_result = run_rag(question, name, 1)

            st.session_state["comparison_result"] = {
                "question": question,
                "person": name,
                "dataset": DATASET,
                "cypher_rows": rows,
                "cypher_latency_ms": cypher_ms,
                "graphrag": rag_result,
            }

        except RuntimeError as exc:
            st.error(str(exc))

    comparison = st.session_state.get("comparison_result")

    if comparison:
        st.caption(comparison["question"])
        left, right = st.columns(2)

        with left:
            st.subheader("Résultats Cypher")
            st.metric(
                "Temps côté client",
                f"{comparison['cypher_latency_ms']:.1f} ms",
            )
            st.dataframe(
                pd.DataFrame(comparison["cypher_rows"]),
                use_container_width=True,
            )

        with right:
            st.subheader("Réponse GraphRAG")
            display_rag(
                comparison["graphrag"],
                "download_comparison_rag",
            )

        download_json(
            "Télécharger la comparaison",
            comparison,
            "comparison.json",
            "download_comparison",
        )

    st.info(
        "Cette vue est une comparaison exploratoire, pas un score "
        "automatique de qualité. GraphRAG récupère au maximum 60 relations "
        "du voisinage ; Cypher renvoie les affiliations correspondant "
        "à sa requête. Une réponse issue du cache ne mesure pas "
        "le temps d’une nouvelle génération."
    )