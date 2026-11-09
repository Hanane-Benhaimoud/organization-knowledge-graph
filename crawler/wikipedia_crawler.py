"""Wikipedia crawler compatible with Wikidata V3 JSON output schema.
Reconstructed from the complete source pasted in the conversation.
"""
import json
import logging
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from crawler.config import (
    HEADERS, WIKIDATA_DIR, WIKIPEDIA_DIR,
    WIKIPEDIA_APIS, WIKIPEDIA_LANGUAGES,
    MAX_DOCUMENTS, MIN_TEXT_LENGTH,
    REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


class WikipediaCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def request(self, language, parameters):
        params = {'format': 'json', 'formatversion': 2, 'maxlag': 5, **parameters}
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(WIKIPEDIA_APIS[language], params=params,
                                            timeout=REQUEST_TIMEOUT)
                if response.status_code in (429, 500, 502, 503, 504):
                    delay = min(90, 5 * (2 ** attempt))
                    retry = response.headers.get('Retry-After')
                    if retry and retry.isdigit():
                        delay = min(300, int(retry))
                    log.warning('Wikipedia HTTP %s, waiting %ss', response.status_code, delay)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                if 'error' in data:
                    error = data['error']
                    code = error.get('code')
                    if code == 'missingtitle':
                        return None
                    if code == 'maxlag':
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(error)
                time.sleep(REQUEST_DELAY)
                return data
            except (requests.RequestException, ValueError) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                log.warning('Retry: %s', exc)
                time.sleep(min(90, 5 * (2 ** attempt)))
        raise RuntimeError('Wikipedia API unavailable')

    def fetch_article(self, language, title):
        data = self.request(language, {
            'action': 'parse', 'page': title, 'prop': 'text|revid|links',
            'redirects': 1, 'disableeditsection': 1,
        })
        return data.get('parse') if data else None

    @staticmethod
    def extract_content(html):
        soup = BeautifulSoup(html, 'html.parser')
        for selector in [
            'script', 'style', 'sup.reference', '.mw-editsection', '.navbox',
            '.toc', '.reflist', '.mw-references-wrap', '.hatnote', '.metadata', 'table',
        ]:
            for tag in soup.select(selector):
                tag.decompose()
        root = soup.select_one('.mw-parser-output') or soup
        sections = []
        heading = 'Introduction'
        buffer = []

        def flush():
            nonlocal buffer
            text = '\n'.join(buffer).strip()
            if text:
                sections.append({'heading': heading, 'text': text})
            buffer = []

        def visit(element):
            nonlocal heading
            for child in element.children:
                name = getattr(child, 'name', None)
                if not name:
                    continue
                if name in ('h2', 'h3', 'h4'):
                    flush()
                    heading = child.get_text(' ', strip=True)
                elif name in ('p', 'ul', 'ol', 'dl'):
                    text = child.get_text(' ', strip=True)
                    if text:
                        buffer.append(text)
                elif name in ('div', 'section'):
                    visit(child)
        visit(root)
        flush()
        full_text = '\n\n'.join(f"{section['heading']}\n{section['text']}" for section in sections)
        return sections, full_text

    @staticmethod
    def extract_references(html):
        soup = BeautifulSoup(html, 'html.parser')
        references, seen = [], set()
        selectors = 'ol.references a.external, .mw-references-wrap a.external'
        for anchor in soup.select(selectors):
            url = anchor.get('href', '')
            if not url.startswith(('https://', 'http://')) or url in seen:
                continue
            seen.add(url)
            references.append({'url': url, 'label': anchor.get_text(' ', strip=True)})
        return references

    @staticmethod
    def extract_links(page, language):
        links, seen = [], set()
        for item in page.get('links', []):
            if item.get('ns', 0) != 0:
                continue
            title = item.get('title') or item.get('*')
            if not title or title in seen:
                continue
            seen.add(title)
            links.append({'title': title, 'url':
                          f'https://{language}.wikipedia.org/wiki/' +
                          quote(title.replace(' ', '_'), safe='')})
        return links

    @staticmethod
    def document_path(qid, language):
        return WIKIPEDIA_DIR / f'{language}_{qid}.json'

    def existing_document(self, qid):
        for language in WIKIPEDIA_LANGUAGES:
            path = self.document_path(qid, language)
            if not path.exists():
                continue
            try:
                data = read_json(path)
                if (data.get('wikidata_id') == qid
                        and data.get('content', {}).get('full_text')
                        and data.get('revision_id')):
                    return data
            except (OSError, ValueError, TypeError):
                continue
        return None

    def collect(self):
        report_path = WIKIDATA_DIR / 'collection_report.json'
        seeds_path = WIKIDATA_DIR / 'wikipedia_seeds.json'
        if not report_path.exists() or not seeds_path.exists():
            raise FileNotFoundError('Run the Wikidata V3 collector first.')
        log.info('DÉBUT collecte Wikipédia : lecture des entrées Wikidata')
        wikidata_report = read_json(report_path)
        if (wikidata_report.get('target_limit_reached')
                or wikidata_report.get('failed_qids')
                or wikidata_report.get('validated_relation_count', 0) == 0):
            raise RuntimeError('Wikidata quality gate failed. Do not start Wikipedia collection.')
        entities = read_json(seeds_path)['entities']
        entities.sort(key=lambda e: (
            not e.get('is_seed', False),
            e.get('entity_type') != 'organization', e['qid'],
        ))
        selected = entities[:MAX_DOCUMENTS]
        collected, existing = 0, 0
        skipped, failures = [], []
        log.info('Targeted crawl: %d selected entities', len(selected))
        for index, entity in enumerate(selected, 1):
            qid = entity['qid']
            if self.existing_document(qid):
                log.info("[%d/%d] %s : document déjà présent", index, len(selected), qid)
                existing += 1
                continue
            pages = entity.get('wikipedia', {})
            available_languages = [lang for lang in WIKIPEDIA_LANGUAGES
                                   if lang in pages and pages[lang].get('title')]
            if not available_languages:
                log.info("[%d/%d] %s : aucune page Wikipédia disponible", index, len(selected), qid)
                skipped.append({'qid': qid, 'reason': 'no_supported_wikipedia_article'})
                continue
            log.info('[%d/%d] %s | %s', index, len(selected), qid, entity.get('label'))
            saved, errors = False, []
            for language in available_languages:
                title = pages[language]['title']
                try:
                    page = self.fetch_article(language, title)
                    if not page:
                        errors.append(f'{language}: missing page')
                        continue
                    html = page.get('text', '')
                    sections, full_text = self.extract_content(html)
                    if len(full_text) < MIN_TEXT_LENGTH:
                        errors.append(f'{language}: insufficient text')
                        continue
                    revision_id = page.get('revid')
                    if not revision_id:
                        errors.append(f'{language}: missing revision')
                        continue
                    canonical_title = page.get('title', title)
                    source_url = (f'https://{language}.wikipedia.org/wiki/' +
                                  quote(canonical_title.replace(' ', '_'), safe=''))
                    document = {
                        'document_id': f'wikipedia_{language}_{qid}',
                        'wikidata_id': qid, 'entity_label': entity.get('label'),
                        'entity_type': entity.get('entity_type'),
                        'seed_categories': entity.get('seed_categories', []),
                        'source': 'wikipedia', 'source_url': source_url,
                        'revision_id': revision_id,
                        'revision_url': f'https://{language}.wikipedia.org/w/index.php?oldid={revision_id}',
                        'pageid': page.get('pageid'), 'title': canonical_title,
                        'language': language, 'collected_at': utc_now(),
                        'content': {'sections': sections, 'full_text': full_text},
                        'references': self.extract_references(html),
                        'outgoing_links': self.extract_links(page, language),
                    }
                    save_json(self.document_path(qid, language), document)
                    collected += 1
                    log.info("[%d/%d] Enregistré %s (%d caractères)", index, len(selected), self.document_path(qid, language).name, len(full_text))
                    saved = True
                    break
                except Exception as exc:
                    errors.append(f'{language}: {exc}')
                    log.warning('Failed %s (%s): %s', qid, language, exc)
            if not saved:
                failures.append({'qid': qid, 'errors': errors})
        report = {
            'finished_at': utc_now(), 'total_candidates': len(entities),
            'selected': len(selected), 'truncated': len(entities) > MAX_DOCUMENTS,
            'new_documents': collected, 'existing_documents': existing,
            'skipped': skipped, 'failed': failures,
        }
        save_json(WIKIPEDIA_DIR / 'crawl_summary.json', report)
        log.info('DONE: %d new, %d existing, %d skipped, %d failed',
                 collected, existing, len(skipped), len(failures))
        if failures:
            log.warning('Some documents were not collected. Inspect crawl_summary.json')


if __name__ == '__main__':
    WikipediaCrawler().collect()
