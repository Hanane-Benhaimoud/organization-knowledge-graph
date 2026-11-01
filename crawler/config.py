import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()

if not CONTACT_EMAIL or "@" not in CONTACT_EMAIL:
    raise ValueError("Configure CONTACT_EMAIL dans .env")

HEADERS = {
    "User-Agent": (
        f"OrganizationKnowledgeGraph/3.0 "
        f"(contact: {CONTACT_EMAIL})"
    ),
    "Accept": "application/json",
}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

WIKIPEDIA_APIS = {
    "en": "https://en.wikipedia.org/w/api.php",
    "fr": "https://fr.wikipedia.org/w/api.php",
}

BRONZE = ROOT / "data" / "bronze"
WIKIDATA_DIR = BRONZE / "wikidata"
WIKIPEDIA_DIR = BRONZE / "wikipedia"

for path in [WIKIDATA_DIR, WIKIPEDIA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Corpus pilote : organisations explicitement choisies.
# Les QID seront revérifiés auprès de Wikidata.
SEED_ORGANIZATIONS = {
    "technology": [
        "Q95",        # Google
        "Q20800404",  # Alphabet
        "Q15733006",  # Google DeepMind
        "Q15330",     # Waymo
    ],
    "industry": [
        "Q29637",     # Skoda Auto
        "Q156578",    # Volkswagen Group
        "Q29803",     # Isuzu
        "Q53247",     # Panasonic
        "Q53257",     # Mitsubishi Electric
        "Q44294",     # Ford
    ],
    "media": [
        "Q11148",     # The Guardian
        "Q2665980",   # Guardian Media Group
        "Q163560",    # Axel Springer
        "Q29872",     # Die Welt
        "Q284183",    # de Volkskrant
        "Q284947",    # Christian Science Monitor
    ],
    "nonprofit": [
        "Q180",       # Wikimedia Foundation
        "Q858851",    # Linux Foundation
        "Q1377855",   # Google.org
        "Q182380",    # Fair Labor Association
    ],
}

# Wikidata properties that are useful for our graph.
# Relation direction always follows Wikidata's statement.
RELATIONS = {
    "P169": ("CEO", "person"),
    "P112": ("FOUNDED_BY", "person"),
    "P488": ("CHAIRPERSON", "person"),
    "P3320": ("BOARD_MEMBER", "person"),
    "P749": ("PARENT_ORGANIZATION", "organization"),
    "P355": ("SUBSIDIARY", "organization"),
    "P127": ("OWNED_BY", "organization"),
    "P463": ("MEMBER_OF", "organization"),
    "P123": ("PUBLISHER", "organization"),
}

# Wikidata type identifiers
HUMAN_QID = "Q5"
ORGANIZATION_ROOTS = {
    "Q43229",    # organization
    "Q4830453",  # business
    "Q11032",    # newspaper
    "Q163740",   # nonprofit organization
}

# A single hop is deliberate.
MAX_GRAPH_DEPTH = 1

# Safety bounds. Reaching a bound is reported as truncation.
MAX_TARGETS = 1200
MAX_TYPE_NODES = 2500
MAX_TYPE_DEPTH = 6

WIKIDATA_BATCH_SIZE = 40

# One article per QID, English preferred for LLM extraction.
WIKIPEDIA_LANGUAGES = ("en", "fr")
MAX_DOCUMENTS = 300
MIN_TEXT_LENGTH = 300

REQUEST_TIMEOUT = 45
REQUEST_DELAY = 1.2
MAX_RETRIES = 4
