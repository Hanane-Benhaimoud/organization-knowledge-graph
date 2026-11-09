import json
import logging
import time
from collections import Counter, deque
from datetime import datetime, timezone

import requests

from crawler.config import (
    HEADERS, WIKIDATA_API, WIKIDATA_DIR, SEED_ORGANIZATIONS, RELATIONS,
    HUMAN_QID, ORGANIZATION_ROOTS, MAX_GRAPH_DEPTH, MAX_TARGETS,
    MAX_TYPE_NODES, MAX_TYPE_DEPTH, WIKIDATA_BATCH_SIZE,
    REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(name, payload):
    path = WIKIDATA_DIR / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def qid_from_snak(snak):
    value = (snak.get('datavalue') or {}).get('value')
    if isinstance(value, dict):
        qid = value.get('id')
        if isinstance(qid, str) and qid.startswith('Q'):
            return qid
    return None


def resolve_entity_label(raw_entity, qid):
    """English/French label, then Wikipedia sitelink, then any label, then QID."""
    labels = raw_entity.get('labels') or {}
    sitelinks = raw_entity.get('sitelinks') or {}
    for lang in ('en', 'fr'):
        value = (labels.get(lang) or {}).get('value')
        if isinstance(value, str) and value.strip():
            return value.strip()
    for site in ('enwiki', 'frwiki'):
        title = (sitelinks.get(site) or {}).get('title')
        if isinstance(title, str) and title.strip():
            return title.strip()
    for item in labels.values():
        if isinstance(item, dict):
            value = item.get('value')
            if isinstance(value, str) and value.strip():
                return value.strip()
    return qid


def description(raw):
    descriptions = raw.get('descriptions') or {}
    for lang in ('en', 'fr'):
        value = (descriptions.get(lang) or {}).get('value')
        if value:
            return value
    return None


def instance_of(raw):
    return sorted({qid for stmt in (raw.get('claims') or {}).get('P31', [])
                   if (qid := qid_from_snak(stmt.get('mainsnak') or {}))})


def wikipedia_pages(raw):
    links = raw.get('sitelinks') or {}
    return {lang: {'title': links[lang + 'wiki']['title'], 'url': None}
            for lang in ('en', 'fr')
            if isinstance(links.get(lang + 'wiki'), dict)
            and links[lang + 'wiki'].get('title')}


class WikidataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.raw = {}
        self.type_cache = {}
        self.failed = set()
        self.hierarchy_incomplete = False
        self.target_limit_reached = False

    def request(self, params):
        label = params.get("ids", "?")
        log.debug("API Wikidata: %s", label)
        parameters = {'action': 'wbgetentities', 'format': 'json', 'formatversion': 2,
                      'props': 'labels|descriptions|claims|sitelinks', 'languages': 'en|fr',
                      'languagefallback': 0, 'maxlag': 5, **params}
        for attempt in range(MAX_RETRIES):
            if attempt:
                log.warning("Nouvelle tentative API %d/%d : %s", attempt + 1, MAX_RETRIES, label)
            try:
                response = self.session.get(WIKIDATA_API, params=parameters,
                                            timeout=REQUEST_TIMEOUT)
                if response.status_code in (429, 500, 502, 503, 504):
                    retry = response.headers.get('Retry-After', '')
                    delay = min(300, int(retry)) if retry.isdigit() else min(90, 5 * 2 ** attempt)
                    log.warning('HTTP %s pour %s : attente %ss', response.status_code, label, delay)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                result = response.json()
                if 'error' in result:
                    if result['error'].get('code') == 'maxlag':
                        log.warning('Wikidata maxlag : %s', label)
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(result['error'])
                time.sleep(REQUEST_DELAY)
                return result
            except (requests.RequestException, ValueError) as exc:
                log.warning("Erreur API %s : %s", label, exc)
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(min(90, 5 * 2 ** attempt))
        raise RuntimeError('Wikidata API unavailable')

    def fetch(self, qids):
        missing = sorted(set(qids) - set(self.raw) - self.failed)
        log.info("Récupération de %d entités manquantes (lots de %d)", len(missing), WIKIDATA_BATCH_SIZE)
        for i in range(0, len(missing), WIKIDATA_BATCH_SIZE):
            batch = missing[i:i + WIKIDATA_BATCH_SIZE]
            log.info("Lot %d/%d : %d QID (%s ... %s)", i // WIKIDATA_BATCH_SIZE + 1, (len(missing) + WIKIDATA_BATCH_SIZE - 1) // WIKIDATA_BATCH_SIZE, len(batch), batch[0], batch[-1])
            try:
                result = self.request({'ids': '|'.join(batch)})
                for qid, raw in (result.get('entities') or {}).items():
                    if raw.get('missing') or raw.get('type') != 'item':
                        self.failed.add(qid)
                    else:
                        self.raw[qid] = raw
                self.failed.update(set(batch) - set((result.get('entities') or {})))
            except Exception as exc:
                log.error('Failed Wikidata batch: %s', exc)
                self.failed.update(batch)

    def relation_records(self, seed_qids, timestamp):
        records = []
        for subject in sorted(seed_qids):
            raw = self.raw.get(subject)
            if not raw:
                continue
            for property_id, (predicate, expected_type) in RELATIONS.items():
                for statement in (raw.get('claims') or {}).get(property_id, []):
                    target = qid_from_snak(statement.get('mainsnak') or {})
                    if not target:
                        continue
                    records.append({
                        'statement_id': statement.get('id'), 'subject': subject,
                        'predicate': predicate, 'property_id': property_id,
                        'object': target, 'expected_object_type': expected_type,
                        'rank': statement.get('rank', 'normal'),
                        'qualifiers': statement.get('qualifiers') or {},
                        'references': statement.get('references') or [],
                        'source': 'wikidata',
                        'source_url': f'https://www.wikidata.org/wiki/{subject}',
                        'retrieved_at': timestamp,
                    })
        return records

    def organization_type_ids(self, raw_types):
        """Resolve P279 ancestors with a bounded BFS; incomplete results stay unverified."""
        roots = set(ORGANIZATION_ROOTS)
        queue = deque((qid, 0) for qid in sorted(raw_types - roots - {HUMAN_QID}))
        seen = set()
        parents = {}
        log.info("Validation hiérarchique : %d types de départ", len(queue))
        while queue:
            qid, depth = queue.popleft()
            if qid in seen:
                continue
            if len(seen) >= MAX_TYPE_NODES or depth > MAX_TYPE_DEPTH:
                self.hierarchy_incomplete = True
                continue
            seen.add(qid)
            if len(seen) == 1 or len(seen) % 25 == 0:
                log.info("Hiérarchie P279 : %d types examinés, %d en attente, profondeur %d", len(seen), len(queue), depth)
            if qid not in self.type_cache:
                # Type-hierarchy lookups must NOT be included in raw_entities.json.
                try:
                    response = self.request({'ids': qid})
                    self.type_cache[qid] = (response.get('entities') or {}).get(qid)
                except Exception:
                    self.type_cache[qid] = None
            raw = self.type_cache.get(qid)
            if not raw:
                self.hierarchy_incomplete = True
                continue
            p = {x for stmt in (raw.get('claims') or {}).get('P279', [])
                 if (x := qid_from_snak(stmt.get('mainsnak') or {}))}
            parents[qid] = p
            if depth < MAX_TYPE_DEPTH:
                queue.extend((x, depth + 1) for x in sorted(p - seen - roots))
            elif p - roots:
                self.hierarchy_incomplete = True

        memo = {}
        def is_org(qid, trail=frozenset()):
            if qid in roots:
                return True
            if qid in trail:
                return False
            if qid in memo:
                return memo[qid]
            result = any(is_org(p, trail | {qid}) for p in parents.get(qid, ()))
            memo[qid] = result
            return result
        log.info("Hiérarchie terminée : %d types examinés, incomplète=%s", len(seen), self.hierarchy_incomplete)
        return {qid for qid in raw_types if is_org(qid)} | roots

    def collect(self):
        started = utc_now()
        log.info("DÉBUT collecte Wikidata")
        seed_categories = {}
        for category, qids in SEED_ORGANIZATIONS.items():
            for qid in qids:
                seed_categories.setdefault(qid, set()).add(category)
        seeds = set(seed_categories)
        log.info("Étape 1/6 : collecte des %d organisations initiales", len(seeds))
        self.fetch(seeds)
        log.info("Étape 2/6 : extraction des relations des organisations initiales")
        raw_relations = self.relation_records(seeds, started)
        targets = sorted({r['object'] for r in raw_relations} | seeds)
        if len(targets) > MAX_TARGETS:
            self.target_limit_reached = True
            targets = targets[:MAX_TARGETS]
        log.info("%d relations brutes ; %d cibles distinctes", len(raw_relations), len(targets))
        log.info("Étape 3/6 : collecte des entités cibles")
        self.fetch(targets)
        log.info("Étape 4/6 : vérification des types P31/P279 (potentiellement longue)")
        raw_types = {t for qid in targets if qid in self.raw for t in instance_of(self.raw[qid])}
        organization_types = self.organization_type_ids(raw_types)

        def classify(qid):
            raw = self.raw.get(qid)
            if not raw:
                return 'unknown'
            types = set(instance_of(raw))
            if HUMAN_QID in types:
                return 'person'
            if types & organization_types or qid in seeds:
                return 'organization'
            return 'unknown'

        log.info("Étape 5/6 : validation des relations")
        valid, rejected = [], []
        for record in raw_relations:
            reasons = []
            if record['rank'] == 'deprecated':
                reasons.append('deprecated_statement')
            if record['object'] not in self.raw or classify(record['object']) != record['expected_object_type']:
                reasons.append('target_type_not_verified')
            if reasons:
                rejected.append({**record, 'rejection_reasons': reasons})
            else:
                valid.append(record)

        graph_qids = sorted(seeds | {q for r in valid for q in (r['subject'], r['object'])})
        def entity_record(qid):
            raw = self.raw[qid]
            return {
                'qid': qid, 'label': resolve_entity_label(raw, qid),
                'description': description(raw), 'instance_of': instance_of(raw),
                'is_seed': qid in seeds,
                'seed_categories': sorted(seed_categories.get(qid, ())),
                'wikipedia': wikipedia_pages(raw),
                'source': 'wikidata',
                'source_url': f'https://www.wikidata.org/wiki/{qid}',
                'entity_type': classify(qid),
            }
        entities = [entity_record(qid) for qid in graph_qids if qid in self.raw]
        wikipedia_candidates = [e for e in entities if e['wikipedia']]
        log.info('Étape 6/6 : écriture des JSON (%d entités, %d relations validées, %d candidats Wikipédia)', len(entities), len(valid), len(wikipedia_candidates))
        stamp = started
        save_json('seeds.json', {'collected_at': stamp, 'count': len(seeds),
                                 'seeds': [{'qid': q, 'categories': sorted(seed_categories[q])}
                                           for q in sorted(seeds)]})
        save_json('raw_entities.json', {'collected_at': stamp, 'entities': dict(sorted(self.raw.items()))})
        for name, rows in [('raw_relations.json', raw_relations),
                           ('relations.json', valid), ('rejected_relations.json', rejected)]:
            save_json(name, {'collected_at': stamp, 'count': len(rows), 'relations': rows})
        save_json('entities.json', {'collected_at': stamp, 'count': len(entities), 'entities': entities})
        save_json('wikipedia_seeds.json', {'collected_at': stamp, 'count': len(wikipedia_candidates),
                                           'entities': wikipedia_candidates})
        report = {
            'started_at': started, 'finished_at': utc_now(), 'seed_count': len(seeds),
            'raw_relation_count': len(raw_relations), 'validated_relation_count': len(valid),
            'rejected_relation_count': len(rejected), 'graph_entity_count': len(entities),
            'wikipedia_candidate_count': len(wikipedia_candidates),
            'total_target_count': len(targets), 'target_limit_reached': self.target_limit_reached,
            'type_hierarchy_incomplete': self.hierarchy_incomplete,
            'failed_qids': sorted(self.failed),
            'entity_types': dict(Counter(classify(qid) for qid in targets if qid in self.raw)),
        }
        save_json('collection_report.json', report)
        log.info('Done: %d entities, %d relations, %d Wikipedia seeds',
                 len(entities), len(valid), len(wikipedia_candidates))
        return report


if __name__ == '__main__':
    WikidataCollector().collect()
