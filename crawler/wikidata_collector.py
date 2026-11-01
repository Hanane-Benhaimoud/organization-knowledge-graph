
def resolve_entity_label(raw_entity, qid):
    """
    Resolve a human-readable label from Wikidata.

    Priority:
    1. English Wikidata label
    2. French Wikidata label
    3. English Wikipedia title
    4. French Wikipedia title
    5. Other available Wikidata labels
    6. QID as final fallback
    """
    labels = raw_entity.get("labels") or {}
    sitelinks = raw_entity.get("sitelinks") or {}

    for lang in ("en", "fr"):
        value = (labels.get(lang) or {}).get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()

    for site in ("enwiki", "frwiki"):
        title = (sitelinks.get(site) or {}).get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

    for item in labels.values():
        if isinstance(item, dict):
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()

    return qid
