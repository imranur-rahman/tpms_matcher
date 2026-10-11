# tpms_matcher.py

import io
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .dblp_local import DEFAULT_INDEX_PATH, DblpIndex, fold_name

# ----------- Config ----------- #
DBLP_AUTHOR_SEARCH_API = "https://dblp.org/search/author/api"
DBLP_PERSON_XML = "https://dblp.org/pid/{}.xml"
OPENALEX_AUTHORS_API = "https://api.openalex.org/authors"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_AUTHOR_SEARCH = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR_PAPERS = "https://api.semanticscholar.org/graph/v1/author/{}/papers"
CROSSREF_WORKS_API = "https://api.crossref.org/works/{}"
DECAY_HALF_LIFE = 5  # years
CURRENT_YEAR = datetime.now().year
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 5
# Smallest time between two requests to one API host, shared by all worker threads.
# Semantic Scholar allows 1 request per second; the others ask for polite clients.
HOST_MIN_INTERVAL = {
    "api.semanticscholar.org": 1.0,
    "api.openalex.org": 0.1,
    "api.crossref.org": 0.2,
    "dblp.org": 1.0,
}
API_CACHE_TTL = 7 * 24 * 3600  # seconds
# A longer Retry-After means a used-up daily budget; waiting is no use in this run
MAX_RETRY_WAIT = 120  # seconds
REQUEST_HEADERS = {
    "User-Agent": "tpms_matcher/0.1 (+https://github.com/imranur-rahman/tpms_matcher)"
}
MAX_OPENALEX_WORKS = 1000
# OpenAlex work types that are not research papers (data sets, reviews, front matter)
OPENALEX_SKIPPED_TYPES = {
    "dataset", "paratext", "peer-review", "erratum", "editorial", "letter",
    "grant", "supplementary-materials", "retraction", "other",
}
MAX_SEMANTIC_SCHOLAR_PAPERS = 1000
AUTHOR_CANDIDATES = 10
# HotCRP lists the PC at <site>/users/pc; the show[] params add the optional columns
HOTCRP_PC_PATH = "users/pc"
HOTCRP_PC_PARAMS = {"show[]": ["orcid", "country"]}
ORCID_PATTERN = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]")
# Every publication document gets about the same length, so that short
# documents (titles) and long documents (full PDFs) score on the same scale
DOC_WORD_LIMIT = 400
PDF_WORD_LIMIT = 3000
SPECTER_MODEL = "sentence-transformers/allenai-specter"
LSA_COMPONENTS = 200
DEFAULT_TOP_K = 3
DEFAULT_ALPHA = 0.5
DEFAULT_COI_YEARS = 2
UNKNOWN_YEAR_WEIGHT = 0.5
PARALLEL_WORKERS = 4
# researchr bans clients that fetch profiles too fast, so profiles load one at a time.
# The tool waits at least PROFILE_DELAY, or the Crawl-delay of robots.txt when it is longer.
PROFILE_DELAY = 2.0
DEFAULT_PROFILE_TOP = 20
# Identity methods that point to one person; "unique-name" and "name-only" do not
CONFIDENT_IDENTITY_METHODS = {"orcid", "profile-link", "affiliation"}
HTML_CACHE_DIR = Path(os.environ.get("TPMS_CACHE_DIR", Path.home() / ".cache" / "tpms_matcher")) / "html"
HTML_CACHE_TTL = 7 * 24 * 3600  # seconds
PDF_CACHE_DIR = HTML_CACHE_DIR.parent / "pdf_text"
API_CACHE_DIR = HTML_CACHE_DIR.parent / "api"
# The abstract is on the first page; two pages leave room for long author lists
ABSTRACT_PAGES = 2
# Hard limits, so one huge or broken PDF cannot stop the run
PDF_MAX_BYTES = 20 * 1024 * 1024
PDF_MAX_PAGES = 30
BLOCK_PAGE_MARKERS = ["CrowdSec Captcha", "Protected by <a href=\"https://github.com/TecharoHQ/anubis\"", "cf-challenge", "<title>Just a moment...</title>"]

PDF_TEXT_CACHE = {}
PDF_CACHE_LOCK = threading.Lock()
DBLP_API_STATE = {"blocked": False}
# Sources that ran out of budget in this run, with the advice to print once
EXHAUSTED_SOURCES = {}
SOURCE_ADVICE = {
    "openalex": "Get a free key at https://openalex.org/settings/api and set OPENALEX_API_KEY.",
    "semanticscholar": "Get a key at https://www.semanticscholar.org/product/api and set SEMANTIC_SCHOLAR_API_KEY.",
}


def mark_exhausted(source, error):
    with PDF_CACHE_LOCK:
        if source in EXHAUSTED_SOURCES:
            return
        EXHAUSTED_SOURCES[source] = str(error)
    print(
        f"{error}. The tool skips {source} for the rest of this run. "
        f"{SOURCE_ADVICE.get(source, '')} Cached answers stay usable."
    )
SOURCE_FAILURES = {}


def count_failure(source):
    with PDF_CACHE_LOCK:
        SOURCE_FAILURES[source] = SOURCE_FAILURES.get(source, 0) + 1

# ----------- Utility Functions ----------- #

@contextmanager
def suppress_pdf_warnings():
    noisy_loggers = [
        "pdfminer",
        "pdfminer.pdfparser",
        "pdfminer.pdfpage",
        "pdfminer.pdftypes",
        "pdfminer.pdfinterp",
        "pdfminer.converter",
    ]
    previous_levels = {}
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        previous_levels[logger_name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        with redirect_stderr(io.StringIO()):
            yield
    finally:
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)

def exponential_decay_weight(year, current_year=CURRENT_YEAR, half_life=DECAY_HALF_LIFE):
    if not half_life or half_life <= 0:
        return 1.0
    if not year:
        return UNKNOWN_YEAR_WEIGHT
    age = max(0, current_year - int(year))
    return math.exp(-math.log(2) * age / half_life)


class NotJsonResponse(Exception):
    """The server answered with HTML (for example a bot check page) instead of JSON"""


class RateLimitExhausted(Exception):
    """The server asks for a wait that is too long for this run (for example a daily budget)"""


HOST_NEXT_REQUEST = {}
HOST_LOCK = threading.Lock()


def wait_for_host(url):
    """Space the requests to one host by HOST_MIN_INTERVAL, across all threads"""
    host = urlsplit(url).netloc.lower()
    interval = HOST_MIN_INTERVAL.get(host)
    if not interval:
        return
    with HOST_LOCK:
        now = time.monotonic()
        start = max(now, HOST_NEXT_REQUEST.get(host, 0.0))
        HOST_NEXT_REQUEST[host] = start + interval
    if start > now:
        time.sleep(start - now)


def http_get(url, params=None, headers=None):
    """GET with a rate limit for each host and retries on rate limits and server errors"""
    all_headers = dict(REQUEST_HEADERS)
    all_headers.update(headers or {})
    for attempt in range(REQUEST_RETRIES + 1):
        wait_for_host(url)
        resp = requests.get(url, headers=all_headers, timeout=REQUEST_TIMEOUT, params=params)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < REQUEST_RETRIES:
            retry_after = resp.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            if wait > MAX_RETRY_WAIT:
                hours = wait / 3600
                raise RateLimitExhausted(
                    f"{urlsplit(url).netloc} asks to wait {hours:.1f} hours (HTTP {resp.status_code})"
                )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return resp


class BlockedPage(Exception):
    """The site answered with a captcha or bot check page instead of content"""


def is_block_page(html):
    return any(marker in html[:20000] for marker in BLOCK_PAGE_MARKERS)


def html_cache_path(url):
    return HTML_CACHE_DIR / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html")


def fetch_html(url, cache=False):
    """GET a page. With cache=True, reuse a copy younger than HTML_CACHE_TTL."""
    path = html_cache_path(url)
    if cache and path.exists() and time.time() - path.stat().st_mtime < HTML_CACHE_TTL:
        return path.read_text(encoding="utf-8")
    html = http_get(url).text
    if is_block_page(html):
        raise BlockedPage(f"{urlsplit(url).netloc} answered with a captcha or bot check page")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html


def read_page(location, cache=True):
    """Read a page from a URL, or from a local HTML file that the user saved"""
    if os.path.isfile(location):
        return Path(location).read_text(encoding="utf-8", errors="replace")
    return fetch_html(location, cache=cache)


def api_cache_path(url, params):
    key = json.dumps([url, sorted((params or {}).items())], sort_keys=True, default=str)
    return API_CACHE_DIR / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")


def fetch_json(url, params=None, headers=None):
    """GET JSON. Answers are cached on disk for API_CACHE_TTL, so a repeated run sends few requests."""
    path = api_cache_path(url, params)
    if path.exists() and time.time() - path.stat().st_mtime < API_CACHE_TTL:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    resp = http_get(url, params=params, headers=headers)
    try:
        data = resp.json()
    except ValueError:
        raise NotJsonResponse(f"{url} did not return JSON ({resp.headers.get('Content-Type')})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def normalize_title(title):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def normalize_orcid(value):
    """Return the bare ORCID iD from a URL or an iD string, or an empty string"""
    match = ORCID_PATTERN.search(value or "")
    return match.group().upper() if match else ""


def names_match(name_a, name_b):
    """Compare person names without accents, case, and DBLP homonym numbers.

    "J. Smith" matches "John Smith", and "John A. Smith" matches "John Smith".
    """
    tokens_a = fold_name(name_a).split()
    tokens_b = fold_name(name_b).split()
    if not tokens_a or not tokens_b:
        return False
    if tokens_a == tokens_b:
        return True
    if tokens_a[-1] != tokens_b[-1]:
        return False
    first_a, first_b = tokens_a[0], tokens_b[0]
    if first_a == first_b:
        return True
    return (len(first_a) == 1 or len(first_b) == 1) and first_a[0] == first_b[0]


AFFILIATION_STOPWORDS = {
    "university", "of", "the", "institute", "technology", "college", "school",
    "department", "dept", "and", "at", "for", "de", "du", "di", "la", "le",
    "center", "centre", "research", "lab", "laboratory", "labs", "inc", "ltd",
    "national", "sciences", "science", "computer", "engineering", "faculty", "in",
}


def affiliation_tokens(text):
    return {t for t in fold_name(text).split() if t not in AFFILIATION_STOPWORDS and len(t) > 1}


def affiliation_similarity(affiliation_a, affiliation_b):
    """Overlap of the distinctive words, from 0 to 1"""
    tokens_a = affiliation_tokens(affiliation_a)
    tokens_b = affiliation_tokens(affiliation_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def member_affiliations(member_info):
    values = [member_info.get("affiliation"), member_info.get("university")]
    return [v for v in dict.fromkeys(values) if v]


def pick_author(candidates, member_info):
    """Choose the candidate that is the PC member.

    Each candidate is a dict with: name, affiliations (list), orcid, count.
    Returns (candidate, method). The method tells how sure the choice is:
    "orcid", "affiliation", "unique-name", or "name-only" (more than one person
    with this name and nothing to tell them apart).
    """
    orcid = normalize_orcid(member_info.get("orcid"))
    if orcid:
        for cand in candidates:
            if normalize_orcid(cand.get("orcid")) == orcid:
                return cand, "orcid"

    name = member_info.get("name", "")
    named = [c for c in candidates if names_match(name, c.get("name", ""))]
    if not named:
        return None, None

    best, best_score = None, 0.0
    for cand in named:
        for member_aff in member_affiliations(member_info):
            for cand_aff in cand.get("affiliations") or []:
                score = affiliation_similarity(member_aff, cand_aff)
                if score > best_score:
                    best, best_score = cand, score
    if best is not None and best_score >= 0.5:
        return best, "affiliation"

    if len(named) == 1:
        return named[0], "unique-name"
    return max(named, key=lambda c: c.get("count") or 0), "name-only"


def merge_publications(existing, incoming):
    for key in ["title", "venue", "url", "pdf_url", "abstract", "authors", "doi", "arxiv_id"]:
        if not existing.get(key) and incoming.get(key):
            existing[key] = incoming[key]
    if incoming.get("abstract") and len(incoming["abstract"]) > len(existing.get("abstract") or ""):
        existing["abstract"] = incoming["abstract"]
    # The preprint and the published version can differ in year; keep the later one
    years = [y for y in (existing.get("year"), incoming.get("year")) if y]
    existing["year"] = max(years) if years else None
    existing_sources = set(existing.get("sources", []))
    incoming_sources = set(incoming.get("sources", []))
    existing["sources"] = sorted(existing_sources | incoming_sources)
    return existing


def publication_doi_key(pub):
    doi = normalize_doi(pub.get("doi"))
    # arXiv DOIs belong to the preprint, so they do not join versions
    if not doi or doi.startswith("10.48550/"):
        return ""
    return doi


def dedupe_publications(publications):
    """Merge the records of one paper from all sources.

    Two records are the same paper when they share a DOI or a normalized
    title. The year is not part of the key, because the preprint and the
    published version often have different years.
    """
    merged = []
    by_title = {}
    by_doi = {}
    for pub in publications:
        pub.setdefault("sources", [])
        title_key = normalize_title(pub.get("title"))
        doi_key = publication_doi_key(pub)
        idx = by_doi.get(doi_key) if doi_key else None
        if idx is None and title_key:
            idx = by_title.get(title_key)
        if idx is None:
            idx = len(merged)
            merged.append(pub)
        else:
            merged[idx] = merge_publications(merged[idx], pub)
        if title_key:
            by_title.setdefault(title_key, idx)
        if doi_key:
            by_doi.setdefault(doi_key, idx)
    return merged


def reconstruct_openalex_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
    position_map = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    return " ".join(position_map[pos] for pos in sorted(position_map))

def is_probable_pdf_link(href, text=""):
    if not href:
        return False
    href_lower = href.lower()
    if href_lower.startswith("mailto:"):
        return False
    if href_lower.endswith(".pdf"):
        return True
    if any(href_lower.endswith(ext) for ext in [".html", ".htm", ".php", ".aspx"]):
        return False
    if "pdf" in href_lower and "pdf" in text.lower():
        return True
    if "pdf" in href_lower and "download" in href_lower:
        return True
    if "pdf" in href_lower and "/pdf" in href_lower:
        return True
    return False


def pdf_cache_path(pdf_url, max_pages):
    digest = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()
    return PDF_CACHE_DIR / f"{digest}.{max_pages or 'all'}.txt"


def download_pdf_text(pdf_url, max_pages=None):
    """Return the text of a PDF. With max_pages, read only the first pages (about 10x faster).

    The text is cached in memory and on disk, so later runs skip the download.
    """
    key = (pdf_url, max_pages)
    with PDF_CACHE_LOCK:
        if key in PDF_TEXT_CACHE:
            return PDF_TEXT_CACHE[key]
    path = pdf_cache_path(pdf_url, max_pages)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        try:
            from pdfminer.high_level import extract_text

            with requests.get(pdf_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, stream=True) as resp:
                resp.raise_for_status()
                content = b""
                for chunk in resp.iter_content(chunk_size=65536):
                    content += chunk
                    if len(content) > PDF_MAX_BYTES:
                        raise ValueError(f"PDF larger than {PDF_MAX_BYTES} bytes")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmpf:
                tmpf.write(content)
                tmpf.flush()
                with suppress_pdf_warnings():
                    text = extract_text(tmpf.name, maxpages=max_pages or PDF_MAX_PAGES) or ""
        except Exception:
            text = ""
        # An empty result is cached too, so a broken link is not tried again
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    with PDF_CACHE_LOCK:
        PDF_TEXT_CACHE[key] = text
    return text

# ----------- Text Cleaning ----------- #

REFERENCE_HEADING = re.compile(
    r"^[#*\s]*(?:\d+\.?|[IVX]+\.)?\s*(references|bibliography|works cited|literature cited)\W*$",
    re.IGNORECASE | re.MULTILINE,
)
INTRODUCTION_HEADING = re.compile(
    r"^[#*\s]*(?:1\.?|I\.)?\s*introduction\W*$", re.IGNORECASE | re.MULTILINE
)
ABSTRACT_START = re.compile(r"^[#*\s]*abstract\b[\s.:—–-]*", re.IGNORECASE | re.MULTILINE)
# Lines from paper templates and publisher footers that carry no topic
BOILERPLATE_LINES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^acm reference format",
        r"permission to make digital or hard copies",
        r"^©|copyright \d{4}|\(c\) \d{4}",
        r"^ccs concepts",
        r"^(additional )?key ?words( and phrases)?\b",
        r"^arxiv:\d",
        r"^https?://\S+$",
        r"^doi:|^https?://doi\.org/",
        r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
        r"^isbn\b|^issn\b",
        r"^publication rights licensed",
        r"^(anonymous author|anonymized|under review|submitted to)",
        r"^\d+$",
    ]
]


def clean_paper_text(text):
    """Remove the reference list and the template text from a paper"""
    if not text:
        return ""
    # Cut at the last reference heading in the second part of the paper
    matches = [m for m in REFERENCE_HEADING.finditer(text) if m.start() > len(text) * 0.3]
    if matches:
        text = text[: matches[-1].start()]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.search(stripped) for p in BOILERPLATE_LINES):
            continue
        lines.append(stripped)
    return normalize_text(" ".join(lines))


def extract_abstract(text, limit=300):
    """Return the abstract of a paper text, or its first words when no abstract heading exists"""
    if not text:
        return ""
    head = text[:20000]
    start = ABSTRACT_START.search(head)
    if start:
        body = head[start.end():]
        end = INTRODUCTION_HEADING.search(body)
        if end:
            body = body[: end.start()]
    else:
        body = head
    return " ".join(clean_paper_text(body).split()[:limit])


def limit_words(text, limit):
    words = (text or "").split()
    return " ".join(words[:limit])


def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def publication_document(pub, text_source):
    """Build the text that represents one publication in the similarity model"""
    title = pub.get("title") or ""
    if text_source == "title":
        return title
    if text_source == "pdf":
        pdf_url = pub.get("pdf_url")
        pdf_text = clean_paper_text(download_pdf_text(pdf_url)) if pdf_url else ""
        if pdf_text:
            return limit_words(f"{title}. {pdf_text}", PDF_WORD_LIMIT)
    abstract = pub.get("abstract") or ""
    if not abstract and text_source in {"auto", "pdf"} and pub.get("pdf_url"):
        abstract = extract_abstract(download_pdf_text(pub["pdf_url"], max_pages=ABSTRACT_PAGES))
    return limit_words(f"{title}. {abstract}" if abstract else title, DOC_WORD_LIMIT)


def needs_pdf(pub, text_source):
    if text_source == "pdf":
        return True
    return text_source == "auto" and not pub.get("abstract")


def run_parallel(func, items, desc, unit):
    items = list(items)
    if not items:
        return []
    try:
        from tqdm import tqdm  # type: ignore
        progress = tqdm(total=len(items), desc=desc, unit=unit)
    except Exception:
        progress = None
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        for result in pool.map(func, items):
            results.append(result)
            if progress:
                progress.update(1)
    if progress:
        progress.close()
    return results


def prefetch_pdf_texts(publications, text_source):
    if text_source not in {"auto", "pdf"}:
        return
    pdf_pubs = [pub for pub in publications if pub.get("pdf_url") and needs_pdf(pub, text_source)]
    print(f"PDF parsing plan: {len(pdf_pubs)}/{len(publications)} publications need a PDF")
    max_pages = None if text_source == "pdf" else ABSTRACT_PAGES
    run_parallel(lambda pub: download_pdf_text(pub["pdf_url"], max_pages), pdf_pubs, "Parsing PDFs", "pdf")

def normalize_doi(doi):
    if not doi:
        return ""
    doi = doi.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"]:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
    return doi.lower()

def resolve_arxiv_pdf(arxiv_id):
    if not arxiv_id:
        return ""
    arxiv_id = arxiv_id.replace("arXiv:", "").strip()
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def resolve_pdf_from_doi(doi):
    doi = normalize_doi(doi)
    if not doi:
        return "", None, "missing_doi"
    if doi.startswith("10.48550/arxiv."):
        arxiv_id = doi.split("10.48550/arxiv.", 1)[1]
        return resolve_arxiv_pdf(arxiv_id), "arxiv", None
    try:
        data = fetch_json(CROSSREF_WORKS_API.format(doi))
        message = data.get("message", {})
        links = message.get("link", [])
        for link in links:
            if link.get("content-type") == "application/pdf":
                return link.get("URL", ""), "crossref", None
    except Exception:
        return "", None, "crossref_error"
    return "", None, "crossref_no_pdf"

def recover_pdf_url(pub):
    if pub.get("pdf_url"):
        return pub["pdf_url"], None
    pdf_url = ""
    doi = pub.get("doi")
    if doi:
        pdf_url, source, failure = resolve_pdf_from_doi(doi)
        if pdf_url:
            return pdf_url, source
    if not pdf_url:
        arxiv_id = pub.get("arxiv_id")
        pdf_url = resolve_arxiv_pdf(arxiv_id) if arxiv_id else ""
        if pdf_url:
            return pdf_url, "arxiv"
    return "", None

def enrich_pdf_urls(publications, text_source="auto"):
    """Find PDF links for the publications that need a PDF for the chosen text source"""
    stats = {
        "existing_by_source": {},
        "recovered_by_source": {},
        "failures_by_source": {},
        "total_publications": 0,
        "total_with_pdf": 0,
    }
    for pub in publications:
        stats["total_publications"] += 1
        if pub.get("pdf_url"):
            stats["total_with_pdf"] += 1
            for src in pub.get("sources", []) or ["unknown"]:
                stats["existing_by_source"][src] = stats["existing_by_source"].get(src, 0) + 1
            continue
        if not needs_pdf(pub, text_source):
            continue
        pdf_url, recovery_source = recover_pdf_url(pub)
        if pdf_url:
            pub["pdf_url"] = pdf_url
            pub["pdf_recovery"] = recovery_source
            stats["total_with_pdf"] += 1
            stats["recovered_by_source"][recovery_source] = (
                stats["recovered_by_source"].get(recovery_source, 0) + 1
            )
        else:
            if (pub.get("doi") or "").strip():
                stats["failures_by_source"]["crossref"] = (
                    stats["failures_by_source"].get("crossref", 0) + 1
                )
            if pub.get("arxiv_id"):
                stats["failures_by_source"]["arxiv"] = (
                    stats["failures_by_source"].get("arxiv", 0) + 1
                )
    return stats

# ----------- Step 1: Crawl CFP Page ----------- #

def extract_profile_details(profile_url):
    """Extract detailed profile information from the profile URL"""
    if not profile_url or "profile" not in profile_url:
        print(f"Invalid profile URL: {profile_url}")
        return {}

    try:
        html = fetch_html(profile_url, cache=True)
        soup = BeautifulSoup(html, "html.parser")

        profile_details = {}

        # Look for profile items
        for item in soup.find_all("div", class_="profile-item"):
            heading_span = item.find("span", class_="profile-item-heading")
            if not heading_span:
                continue

            heading = heading_span.get_text().strip().rstrip(":").lower()

            # Get the content after the heading span
            content = item.get_text().replace(heading_span.get_text(), "").strip()

            # Handle links specially
            link = item.find("a", class_="navigate")
            if link:
                content = link.get("href")

            # Map the headings to our structure
            if heading == "name":
                profile_details["name"] = content
            elif heading == "bio":
                profile_details["bio"] = content
            elif heading == "country":
                profile_details["country"] = content
            elif heading == "affiliation":
                profile_details["affiliation"] = content
            elif heading == "personal website":
                profile_details["personal_website"] = content
            elif heading == "x (twitter)":
                profile_details["twitter"] = content
            elif heading == "research interests":
                profile_details["research_interests"] = content
            elif heading == "linkedin":
                profile_details["linkedin"] = content
            elif heading == "google scholar":
                profile_details["google_scholar"] = content
            elif heading == "orcid":
                profile_details["orcid"] = normalize_orcid(content) or content
            elif heading == "dblp":
                profile_details["dblp"] = content

        return profile_details

    except BlockedPage:
        raise
    except Exception as e:
        print(f"Error fetching profile details from {profile_url}: {e}")
        return {}


ROBOTS_CACHE = {}


def site_rules(url):
    """Return (RobotFileParser or None, delay in seconds) for the site of url"""
    parts = urlsplit(url)
    site = f"{parts.scheme}://{parts.netloc}"
    if site not in ROBOTS_CACHE:
        rules = None
        delay = PROFILE_DELAY
        try:
            text = fetch_html(f"{site}/robots.txt", cache=True)
            rules = RobotFileParser()
            rules.parse(text.splitlines())
            crawl_delay = rules.crawl_delay(REQUEST_HEADERS["User-Agent"]) or rules.crawl_delay("*")
            if crawl_delay:
                delay = max(delay, float(crawl_delay))
        except BlockedPage:
            raise
        except Exception:
            rules = None
        ROBOTS_CACHE[site] = (rules, delay)
    return ROBOTS_CACHE[site]


def add_profile_details(pc_members):
    """Fetch the researchr profile of every member, one at a time, at the site's crawl delay"""
    with_url = [m for m in pc_members if m.get("url") and not m.get("profile_fetched")]
    if not with_url:
        return pc_members
    try:
        rules, delay = site_rules(with_url[0]["url"])
    except BlockedPage as e:
        print(f"{e}. The tool does not fetch profiles.")
        return pc_members
    print(f"Fetching {len(with_url)} researchr profiles, {delay:.0f} s apart")
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(with_url, desc="Fetching profiles", unit="profile")
    except Exception:
        iterator = with_url
    for member in iterator:
        cached = html_cache_path(member["url"]).exists()
        if rules is not None and not rules.can_fetch(REQUEST_HEADERS["User-Agent"], member["url"]):
            continue
        try:
            details = extract_profile_details(member["url"])
        except BlockedPage as e:
            print(
                f"{e}. The tool stops fetching profiles; the members keep the "
                "name, affiliation, and country from the list page."
            )
            break
        # The list page has the display name; keep it if the profile disagrees
        details.pop("name", None)
        member.update({k: v for k, v in details.items() if v})
        member["profile_fetched"] = True
        if not cached:
            time.sleep(delay)
    return pc_members


def is_hotcrp_url(url):
    return bool(url) and "hotcrp" in urlsplit(url).netloc.lower()


def hotcrp_pc_url(hotcrp_url):
    """Build the PC list URL from any page of a HotCRP site.

    Accepts the site root (https://conf.hotcrp.com), the PC page
    (.../users/pc or .../users?t=pc), or any other page on the site.
    Keeps a path prefix for self-hosted instances (https://host/conf26/).
    """
    parts = urlsplit(hotcrp_url.strip())
    # Remove the account selector that HotCRP adds for multiple sign-ins (/u/0/)
    path = re.sub(r"/u/\d+(?=/|$)", "", parts.path)
    for marker in ["/users", "/paper", "/search", "/review", "/profile", "/deadlines", "/index"]:
        idx = path.find(marker)
        if idx != -1:
            path = path[:idx]
            break
    base = urlunsplit((parts.scheme or "https", parts.netloc, path.rstrip("/") + "/", "", ""))
    return urljoin(base, HOTCRP_PC_PATH)


def parse_hotcrp_pc_page(html):
    soup = BeautifulSoup(html, "html.parser")
    pc_members = []
    for row in soup.select("table.pltable tbody tr"):
        name_cell = row.find("td", class_="pl_name")
        if not name_cell:
            continue

        role_span = name_cell.find("span", class_="pcrole")
        role = role_span.get_text().strip() if role_span else None
        if role_span:
            role_span.extract()
        name = re.sub(r"\s+", " ", name_cell.get_text()).strip()
        if not name:
            continue

        def cell_text(css_class):
            cell = row.find("td", class_=css_class)
            return cell.get_text().strip() if cell else ""

        affiliation = cell_text("pl_affiliation") or None
        country = cell_text("pl_country") or None
        orcid = normalize_orcid(cell_text("pl_orcid"))

        member_info = {
            "name": name,
            "url": None,
            "role": role,
            "university": affiliation,
            "affiliation": affiliation,
            "location": country,
            "country": country,
            "source": "hotcrp",
        }
        if orcid:
            member_info["orcid"] = orcid
        pc_members.append(member_info)
    return pc_members


def extract_pc_members_from_hotcrp(hotcrp_url):
    """Extract PC members from the public PC list of a HotCRP site"""
    pc_url = hotcrp_pc_url(hotcrp_url)
    print(f"Fetching PC members from HotCRP: {pc_url}")
    pc_members = parse_hotcrp_pc_page(http_get(pc_url, params=HOTCRP_PC_PARAMS).text)

    if not pc_members:
        print(
            f"No PC members found at {pc_url}. "
            "The PC list of this HotCRP site is probably not public."
        )
    elif len(pc_members) < 5:
        print(
            f"Only {len(pc_members)} PC members found at {pc_url}. "
            "HotCRP probably shows only the chairs to visitors who are not signed in."
        )
    return pc_members


def extract_pc_members(cfp_url, hotcrp_url=None, fetch_profiles=True, max_members=None):
    if is_hotcrp_url(cfp_url):
        return extract_pc_members_from_hotcrp(cfp_url)[:max_members]

    try:
        html = read_page(cfp_url)
    except BlockedPage as e:
        if hotcrp_url:
            print(f"{e}. Falling back to HotCRP.")
            return extract_pc_members_from_hotcrp(hotcrp_url)[:max_members]
        raise BlockedPage(
            f"{e}. Open the page in a browser, save it as HTML, "
            "and pass the file path instead of the URL."
        )
    pc_members = parse_researchr_members(html, cfp_url)
    if not pc_members and hotcrp_url:
        print("No PC members found on the CFP page, falling back to HotCRP.")
        return extract_pc_members_from_hotcrp(hotcrp_url)[:max_members]
    if not pc_members:
        print("No researchr member list found, falling back to generic extraction.")
        pc_members = parse_generic_members(html, cfp_url)
    pc_members = pc_members[:max_members]
    if fetch_profiles:
        add_profile_details(pc_members)
    return pc_members


def parse_researchr_members(html, base_url):
    """Read the member cards of a researchr track page or committee page.

    Track pages put each member in li.list-group-item with h5 headings.
    Committee pages (/committee/...) use div.col-sm-6 cards with h3/h4 headings.
    Both wrap the card in a link to the member profile.
    """
    soup = BeautifulSoup(html, "html.parser")
    members = {}
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if "/profile/" not in href:
            continue
        headings = link.select(".media-heading")
        if not headings:
            continue

        name_heading = headings[0]
        role_elem = name_heading.find("span", class_="pull-right") or name_heading.find("small")
        role = (role_elem.get_text().strip() if role_elem else None) or None
        if role_elem:
            role_elem.extract()
        name = re.sub(r"\s+", " ", name_heading.get_text()).strip()
        if not name:
            continue

        affiliation = None
        country = None
        for heading in headings[1:]:
            if heading.find("span", class_="text-black"):
                affiliation = heading.get_text().strip() or None
            elif heading.find("small"):
                country = heading.get_text().strip() or None
            elif affiliation is None:
                affiliation = heading.get_text().strip() or None

        url = urljoin(base_url, href)
        if url in members:
            # One person can hold two roles on the same page
            if role and role not in (members[url]["role"] or ""):
                members[url]["role"] = ", ".join(filter(None, [members[url]["role"], role]))
            continue
        members[url] = {
            "name": name,
            "url": url,
            "role": role,
            "university": affiliation,
            "affiliation": affiliation,
            "location": country,
            "country": country,
            "source": "researchr",
        }
    return list(members.values())


NAME_PARTICLES = {"van", "von", "de", "der", "den", "da", "di", "del", "della", "la", "le", "du", "bin", "al", "dos", "das"}
NOT_NAME_WORDS = {
    "program", "committee", "track", "call", "papers", "important", "dates", "submission",
    "deadline", "chair", "chairs", "conference", "workshop", "home", "registration",
    "venue", "contact", "university", "institute", "accepted", "keynote", "about", "news",
}


def looks_like_person_name(text):
    tokens = text.split()
    if not 2 <= len(tokens) <= 5 or len(text) > 60:
        return False
    if any(ch.isdigit() for ch in text) or any(ch in text for ch in "@:/|()[]"):
        return False
    for token in tokens:
        lower = token.lower().strip(".,-")
        if lower in NOT_NAME_WORDS:
            return False
        if lower in NAME_PARTICLES:
            continue
        if not token[0].isupper():
            return False
    return True


def parse_generic_members(html, base_url):
    """Fallback for pages without a known layout: keep entries that look like person names"""
    soup = BeautifulSoup(html, "html.parser")
    members = {}
    for tag in soup.find_all(["li", "p", "td"]):
        text = re.sub(r"\s+", " ", tag.get_text()).strip()
        # "Name, Affiliation" and "Name (Affiliation)" are common list formats
        name = re.split(r"\s*[,(–—]\s*| - ", text, maxsplit=1)[0].strip()
        if not looks_like_person_name(name) or name in members:
            continue
        affiliation = text[len(name):].strip(" ,()–—-") or None
        link = tag.find("a", href=True)
        url = urljoin(base_url, link["href"]) if link else None
        members[name] = {
            "name": name,
            "url": url if url and "/profile/" in url else None,
            "university": affiliation,
            "affiliation": affiliation,
            "location": None,
            "role": None,
            "source": "generic",
        }
    return list(members.values())

# ----------- Step 2: External Publication Sources ----------- #

def warn_dblp_blocked(error):
    if not DBLP_API_STATE["blocked"]:
        DBLP_API_STATE["blocked"] = True
        print(
            f"DBLP API is not usable ({error}). The tool skips the DBLP API for this run. "
            "Build a local index from the DBLP dump with --build-dblp-index to use DBLP."
        )


def parse_dblp_person_xml(xml_text):
    """Read all publications from a DBLP person page (https://dblp.org/pid/<pid>.xml)"""
    from lxml import etree

    root = etree.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    publications = []
    for record in root.iterfind("r"):
        for entry in record:
            title_elem = entry.find("title")
            year_elem = entry.find("year")
            if title_elem is None or year_elem is None or not (year_elem.text or "").isdigit():
                continue
            venue_elem = entry.find("booktitle")
            if venue_elem is None:
                venue_elem = entry.find("journal")
            doi = None
            for ee in entry.iterfind("ee"):
                if "doi.org/" in (ee.text or ""):
                    doi = ee.text.split("doi.org/", 1)[1].lower()
                    break
            authors = [re.sub(r"\s+\d{4}$", "", "".join(a.itertext())) for a in entry.iterfind("author")]
            key = entry.get("key", "")
            publications.append(
                {
                    "title": normalize_text("".join(title_elem.itertext())).rstrip("."),
                    "year": int(year_elem.text),
                    "venue": "".join(venue_elem.itertext()) if venue_elem is not None else None,
                    "url": f"https://dblp.org/rec/{key}" if key else "",
                    "authors": authors,
                    "abstract": "",
                    "pdf_url": "",
                    "doi": doi,
                    "arxiv_id": "",
                    "sources": ["dblp"],
                }
            )
    return publications


def dblp_pid_from_url(url):
    match = re.search(r"dblp\.[^/]+/pid/([^?#]+?)(?:\.html|\.xml)?(?:[?#].*)?$", url or "")
    return match.group(1) if match else ""


def fetch_publications_from_dblp_api(member_info):
    pid = dblp_pid_from_url(member_info.get("dblp"))
    method = "profile-link" if pid else None
    if not pid:
        data = fetch_json(
            DBLP_AUTHOR_SEARCH_API,
            params={"q": member_info["name"], "format": "json", "h": AUTHOR_CANDIDATES},
        )
        candidates = []
        for hit in data.get("result", {}).get("hits", {}).get("hit", []):
            info = hit.get("info", {})
            notes = (info.get("notes") or {}).get("note", [])
            if isinstance(notes, dict):
                notes = [notes]
            affiliations = [n.get("text") for n in notes if n.get("@type") == "affiliation"]
            candidates.append(
                {
                    "name": info.get("author", ""),
                    "affiliations": affiliations,
                    "pid": dblp_pid_from_url(info.get("url", "")),
                    "count": 0,
                }
            )
        chosen, method = pick_author(candidates, member_info)
        if not chosen or not chosen.get("pid"):
            return [], None
        pid = chosen["pid"]
    xml_text = http_get(DBLP_PERSON_XML.format(pid)).text
    if not xml_text.lstrip().startswith("<?xml") and "<dblpperson" not in xml_text[:500]:
        raise NotJsonResponse("dblp.org returned a bot check page")
    return parse_dblp_person_xml(xml_text), method


def fetch_publications_from_dblp(member_info, dblp_index=None):
    """Return (publications, identity method) from the local index, or else the DBLP API"""
    if dblp_index is not None:
        person, method = None, None
        if member_info.get("dblp"):
            person = dblp_index.find_by_dblp_url(member_info["dblp"])
            method = "profile-link" if person else None
        if person is None and member_info.get("orcid"):
            person = dblp_index.find_by_orcid(normalize_orcid(member_info["orcid"]))
            method = "orcid" if person else None
        if person is None:
            candidates = [
                {
                    "id": cand["id"],
                    "name": cand["name"],
                    # DBLP joins the current and earlier affiliations with "; "
                    "affiliations": (cand.get("affiliation") or "").split("; ") if cand.get("affiliation") else [],
                    "orcid": cand.get("orcid"),
                    "homepage": cand.get("homepage"),
                    "count": cand["publication_count"],
                }
                for cand in dblp_index.find_by_name(member_info["name"])
            ]
            person, method = pick_author(candidates, member_info)
        if person is None:
            return [], None
        if method != "name-only":
            # The DBLP person record fills what the researchr profile would give
            if person.get("homepage") and not member_info.get("personal_website"):
                member_info["personal_website"] = person["homepage"]
            if person.get("orcid") and not member_info.get("orcid"):
                member_info["orcid"] = person["orcid"]
        return dblp_index.publications(person["id"]), method

    if DBLP_API_STATE["blocked"]:
        return [], None
    try:
        return fetch_publications_from_dblp_api(member_info)
    except NotJsonResponse as e:
        warn_dblp_blocked(e)
    except Exception as e:
        print(f"Error fetching DBLP publications for {member_info['name']}: {e}")
        count_failure("dblp")
    return [], None


def openalex_params(extra):
    params = dict(extra)
    openalex_email = os.environ.get("OPENALEX_EMAIL")
    if openalex_email:
        params["mailto"] = openalex_email
    return params


def openalex_headers():
    # Without a key, OpenAlex gives a small daily budget that all users of one IP share
    api_key = (os.environ.get("OPENALEX_API_KEY") or "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def find_openalex_author(member_info):
    orcid = normalize_orcid(member_info.get("orcid"))
    # An ORCID iD identifies the author exactly; a name search can pick a namesake
    if orcid:
        data = fetch_json(
            OPENALEX_AUTHORS_API,
            params=openalex_params({"filter": f"orcid:{orcid}", "per_page": 1}),
            headers=openalex_headers(),
        )
        results = data.get("results", [])
        if results:
            return results[0].get("id"), "orcid"

    clean_name = re.sub(r"\s+", " ", member_info["name"]).strip()
    data = fetch_json(
        OPENALEX_AUTHORS_API,
        params=openalex_params({"search": clean_name, "per_page": AUTHOR_CANDIDATES}),
        headers=openalex_headers(),
    )
    candidates = []
    for author in data.get("results", []):
        affiliations = [i.get("display_name") for i in author.get("last_known_institutions") or []]
        affiliations += [
            (a.get("institution") or {}).get("display_name") for a in author.get("affiliations") or []
        ]
        candidates.append(
            {
                "id": author.get("id"),
                "name": author.get("display_name", ""),
                "affiliations": [a for a in affiliations if a],
                "orcid": author.get("orcid"),
                "count": author.get("works_count", 0),
            }
        )
    chosen, method = pick_author(candidates, member_info)
    return (chosen["id"], method) if chosen else (None, None)


def fetch_publications_from_openalex(member_info, max_works=MAX_OPENALEX_WORKS):
    publications = []
    method = None
    if "openalex" in EXHAUSTED_SOURCES:
        return publications, None
    try:
        author_id, method = find_openalex_author(member_info)
        if not author_id:
            return publications, None
        cursor = "*"
        while cursor and len(publications) < max_works:
            data = fetch_json(
                OPENALEX_WORKS_API,
                params=openalex_params(
                    {"filter": f"author.id:{author_id}", "per_page": 200, "cursor": cursor}
                ),
                headers=openalex_headers(),
            )
            for work in data.get("results", []):
                if work.get("type") in OPENALEX_SKIPPED_TYPES:
                    continue
                title = work.get("title")
                year = work.get("publication_year")
                source = (work.get("primary_location") or {}).get("source") or {}
                ids = work.get("ids") or {}
                arxiv_id = ids.get("arxiv") if isinstance(ids, dict) else None
                best_oa = work.get("best_oa_location") or {}
                if title and year:
                    publications.append(
                        {
                            "title": title,
                            "year": int(year),
                            "venue": source.get("display_name"),
                            "url": work.get("id"),
                            "authors": [
                                (auth.get("author") or {}).get("display_name")
                                for auth in work.get("authorships", [])
                            ],
                            "abstract": reconstruct_openalex_abstract(work.get("abstract_inverted_index")),
                            "pdf_url": best_oa.get("pdf_url") or "",
                            "doi": work.get("doi"),
                            "arxiv_id": arxiv_id,
                            "sources": ["openalex"],
                        }
                    )
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not data.get("results"):
                break
    except RateLimitExhausted as e:
        mark_exhausted("openalex", e)
        count_failure("openalex")
    except Exception as e:
        print(f"Error fetching OpenAlex publications for {member_info['name']}: {e}")
        count_failure("openalex")

    return publications, method


def semantic_scholar_headers():
    api_key = (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
    return {"x-api-key": api_key} if api_key else {}


def fetch_publications_from_semantic_scholar(member_info, max_papers=MAX_SEMANTIC_SCHOLAR_PAPERS):
    publications = []
    method = None
    if "semanticscholar" in EXHAUSTED_SOURCES:
        return publications, None
    try:
        data = fetch_json(
            SEMANTIC_SCHOLAR_AUTHOR_SEARCH,
            params={
                "query": member_info["name"],
                "limit": AUTHOR_CANDIDATES,
                "fields": "authorId,name,affiliations,paperCount,externalIds",
            },
            headers=semantic_scholar_headers(),
        )
        candidates = [
            {
                "id": a.get("authorId"),
                "name": a.get("name", ""),
                "affiliations": a.get("affiliations") or [],
                "orcid": (a.get("externalIds") or {}).get("ORCID"),
                "count": a.get("paperCount", 0),
            }
            for a in data.get("data", [])
        ]
        chosen, method = pick_author(candidates, member_info)
        if not chosen or not chosen.get("id"):
            return publications, None
        fields = "title,year,venue,url,authors,openAccessPdf,externalIds,abstract"
        offset = 0
        while offset is not None and offset < max_papers:
            page = fetch_json(
                SEMANTIC_SCHOLAR_AUTHOR_PAPERS.format(chosen["id"]),
                params={"fields": fields, "limit": 100, "offset": offset},
                headers=semantic_scholar_headers(),
            )
            for paper in page.get("data", []):
                title = paper.get("title")
                year = paper.get("year")
                external_ids = paper.get("externalIds") or {}
                if title and year:
                    publications.append(
                        {
                            "title": title,
                            "year": int(year),
                            "venue": paper.get("venue"),
                            "url": paper.get("url"),
                            "authors": [a.get("name") for a in paper.get("authors", []) if a.get("name")],
                            "abstract": paper.get("abstract") or "",
                            "pdf_url": (paper.get("openAccessPdf") or {}).get("url") or "",
                            "doi": external_ids.get("DOI"),
                            "arxiv_id": external_ids.get("ArXiv"),
                            "sources": ["semanticscholar"],
                        }
                    )
            offset = page.get("next")
    except RateLimitExhausted as e:
        mark_exhausted("semanticscholar", e)
        count_failure("semanticscholar")
    except Exception as e:
        print(f"Error fetching Semantic Scholar publications for {member_info['name']}: {e}")
        count_failure("semanticscholar")

    return publications, method

# ----------- Step 3: Parse Reviewers Website ----------- #

PUBLICATION_PAGE_PATTERN = re.compile(r"\b(publications?|papers|pubs|bibliography)\b", re.IGNORECASE)
MAX_PUBLICATION_PAGES = 3


def site_host(url):
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def same_site(url_a, url_b):
    host_a = site_host(url_a)
    return bool(host_a) and host_a == site_host(url_b)


def fetch_publications_from_website(member_info):
    """Crawl personal website to extract PDF and BibTeX files"""
    personal_website = member_info.get("personal_website")
    if not personal_website:
        return {"pdfs": [], "bibtex": [], "publications": []}

    try:
        html = fetch_html(personal_website, cache=True)
        soup = BeautifulSoup(html, "html.parser")
        pages = [(personal_website, soup)]

        # Follow only the publication pages of the same site
        pub_urls = []
        for link in soup.find_all("a", href=True):
            target = urljoin(personal_website, link["href"]).split("#")[0]
            label = f"{link.get_text()} {urlsplit(target).path}"
            if (
                target != personal_website
                and same_site(target, personal_website)
                and PUBLICATION_PAGE_PATTERN.search(label)
                and not target.lower().endswith((".pdf", ".bib"))
                and target not in pub_urls
            ):
                pub_urls.append(target)
        for pub_url in pub_urls[:MAX_PUBLICATION_PAGES]:
            try:
                pages.append((pub_url, BeautifulSoup(fetch_html(pub_url, cache=True), "html.parser")))
            except Exception as e:
                print(f"Error crawling publication page {pub_url}: {e}")

        pdfs, bibtex_files, publications = [], [], []
        seen_links = set()
        for page_url, page_soup in pages:
            for link in page_soup.find_all("a", href=True):
                href = link.get("href")
                full_url = urljoin(page_url, href)
                if full_url in seen_links:
                    continue
                seen_links.add(full_url)
                link_text = link.get_text().strip()
                if is_probable_pdf_link(href, link_text):
                    pdfs.append({"url": full_url, "title": link_text, "filename": href.split("/")[-1]})
                elif href.lower().endswith(".bib"):
                    bibtex_files.append({"url": full_url, "title": link_text, "filename": href.split("/")[-1]})
            publications.extend(extract_publications_from_html(page_soup, page_url))

        return {"pdfs": pdfs, "bibtex": bibtex_files, "publications": publications}

    except Exception as e:
        print(f"Error crawling website for {member_info['name']}: {e}")
        return {"pdfs": [], "bibtex": [], "publications": []}


def extract_publications_from_html(soup, base_url):
    """Extract publication information from HTML content"""
    publications = []

    # Look for common publication patterns
    pub_selectors = [
        "div.publication",
        "li.publication",
        "div.paper",
        "li.paper",
        "div.entry",
        "li.entry",
        ".publication-item",
        ".paper-item",
    ]

    for selector in pub_selectors:
        items = soup.select(selector)
        for item in items:
            pub_info = extract_publication_info(item, base_url)
            if pub_info:
                publications.append(pub_info)

    # If no structured publications found, look for patterns in text
    if not publications:
        publications = extract_publications_from_text(soup, base_url)

    return publications


def is_plausible_title(title):
    return bool(title) and 4 <= len(title.split()) <= 40


def extract_publication_info(element, base_url):
    """Extract publication information from a single HTML element"""
    try:
        title = None
        year = None
        venue = None
        pdf_url = None
        bibtex_url = None

        # Try to find title
        title_selectors = [".title", ".paper-title", "strong", "b", "h3", "h4"]
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                break

        # Try to find year (4-digit number)
        text = element.get_text()
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            year = int(year_match.group())

        # Try to find venue
        venue_patterns = [
            r"In\s+([^,\n]+)",
            r"Proceedings of\s+([^,\n]+)",
            r"Conference on\s+([^,\n]+)",
        ]
        for pattern in venue_patterns:
            venue_match = re.search(pattern, text, re.IGNORECASE)
            if venue_match:
                venue = venue_match.group(1).strip()
                break

        # Look for PDF and BibTeX links
        links = element.find_all("a", href=True)
        for link in links:
            href = link.get("href")
            link_text = link.get_text().strip()
            if is_probable_pdf_link(href, link_text):
                pdf_url = urljoin(base_url, href)
            elif href.lower().endswith(".bib") or "bibtex" in href.lower():
                bibtex_url = urljoin(base_url, href)

        if is_plausible_title(title):
            return {
                "title": title,
                "year": year,
                "venue": venue,
                "pdf_url": pdf_url,
                "bibtex_url": bibtex_url,
                "abstract": "",
                "url": "",
                "authors": [],
                "doi": "",
                "arxiv_id": "",
                "sources": ["website"],
            }

    except Exception as e:
        print(f"Error extracting publication info: {e}")

    return None


def extract_publications_from_text(soup, base_url):
    """Extract publications from unstructured text.

    Only list items and paragraphs that have a year and a link count, and the
    caller keeps an entry only when its title matches a paper from an API.
    """
    publications = []

    for element in soup.find_all(["p", "li"]):
        text = normalize_text(element.get_text())
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if not year_match or len(text.split()) <= 5 or not element.find("a", href=True):
            continue

        # A quoted title is the most reliable; else the first sentence
        quoted = re.search(r"[\"“]([^\"”]{15,300})[\"”]", text)
        first_sentence = re.match(r"^([^.]+)\.", text)
        if quoted:
            title = quoted.group(1)
        elif first_sentence:
            title = first_sentence.group(1)
        else:
            title = ""
        title = title.strip(" .,")
        if not is_plausible_title(title):
            continue

        pdf_url = None
        bibtex_url = None
        for link in element.find_all("a", href=True):
            href = link.get("href")
            link_text = link.get_text().strip()
            if is_probable_pdf_link(href, link_text):
                pdf_url = urljoin(base_url, href)
            elif href.lower().endswith(".bib"):
                bibtex_url = urljoin(base_url, href)

        publications.append(
            {
                "title": title,
                "year": int(year_match.group()),
                "venue": None,
                "pdf_url": pdf_url,
                "bibtex_url": bibtex_url,
                "abstract": "",
                "url": "",
                "authors": [],
                "doi": "",
                "arxiv_id": "",
                "sources": ["website"],
            }
        )

    return publications


def download_bibtex_content(bibtex_url):
    """Download and parse BibTeX content"""
    try:
        return http_get(bibtex_url).text
    except Exception as e:
        print(f"Error downloading BibTeX from {bibtex_url}: {e}")
        return None


def _read_braced(text, start, open_char, close_char):
    """Return (content, end index) of a balanced group that starts at text[start] == open_char"""
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "\\":
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return text[start + 1 :], len(text)


def _clean_bibtex_value(value):
    value = re.sub(r"\\[`'\"^~=.uvHckb]\s*\{?(\w)\}?", r"\1", value)  # accents: \"{o} -> o
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    return normalize_text(value)


def _parse_bibtex_fields(body):
    fields = {}
    idx = 0
    while idx < len(body):
        match = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*").match(body, idx)
        if not match:
            break
        name = match.group(1).lower()
        idx = match.end()
        if idx >= len(body):
            break
        if body[idx] == "{":
            value, idx = _read_braced(body, idx, "{", "}")
        elif body[idx] == '"':
            end = idx + 1
            depth = 0
            while end < len(body) and not (body[end] == '"' and depth == 0):
                depth += {"{": 1, "}": -1}.get(body[end], 0)
                end += 1
            value, idx = body[idx + 1 : end], end + 1
        else:
            bare = re.compile(r"[^,\s}]+").match(body, idx)
            value = bare.group() if bare else ""
            idx = bare.end() if bare else idx + 1
        fields[name] = _clean_bibtex_value(value)
    return fields


def parse_bibtex_publications(bibtex_content):
    """Parse BibTeX entries; titles with nested braces such as {{BERT}} are supported"""
    publications = []
    for match in re.finditer(r"@\s*(\w+)\s*([{(])", bibtex_content or ""):
        entry_type = match.group(1).lower()
        if entry_type in {"comment", "string", "preamble"}:
            continue
        open_char = match.group(2)
        body, _ = _read_braced(bibtex_content, match.end() - 1, open_char, "}" if open_char == "{" else ")")
        key_end = body.find(",")
        if key_end == -1:
            continue
        fields = _parse_bibtex_fields(body[key_end + 1 :])
        title = fields.get("title")
        if not title:
            continue
        year_match = re.search(r"\d{4}", fields.get("year", ""))
        arxiv_id = fields.get("eprint", "") if "arxiv" in fields.get("archiveprefix", "").lower() else ""
        publications.append(
            {
                "title": title,
                "year": int(year_match.group()) if year_match else None,
                "venue": fields.get("booktitle") or fields.get("journal"),
                "abstract": fields.get("abstract", ""),
                "url": fields.get("url", ""),
                "authors": [a.strip() for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()],
                "pdf_url": fields.get("pdf", ""),
                "doi": fields.get("doi", ""),
                "arxiv_id": arxiv_id,
                "sources": ["bibtex"],
            }
        )
    return publications


def attach_website_publications(publications, website_publications, website_pdfs):
    """Add PDF links and abstracts from the website to the papers that the APIs found.

    Website entries come from loose HTML heuristics, so the tool keeps an entry
    only when its title matches a known paper. The exception: when the APIs
    found nothing, the website entries are the only evidence and stay.
    """
    if not publications:
        return list(website_publications)

    by_title = {normalize_title(p.get("title")): p for p in publications}
    for entry in website_publications:
        pub = by_title.get(normalize_title(entry.get("title")))
        if pub is None:
            continue
        # The year on a web page is a guess, so only the links and the abstract move over
        for key in ["pdf_url", "abstract", "doi", "arxiv_id"]:
            if entry.get(key) and not pub.get(key):
                pub[key] = entry[key]
        pub["sources"] = sorted(set(pub.get("sources", [])) | set(entry.get("sources", [])))
    for pdf in website_pdfs:
        label = normalize_title(pdf.get("title"))
        pub = by_title.get(label)
        if pub is not None and not pub.get("pdf_url"):
            pub["pdf_url"] = pdf["url"]
            pub["sources"] = sorted(set(pub.get("sources", [])) | {"website"})
    return publications

# ----------- Step 4: Build Reviewer Profiles ----------- #

ARTIFACT_TITLE = re.compile(
    r"\b(reproduction|replication|artifact|artefact)\s+(package|for)\b"
    r"|\((code|data|code and data|dataset|artifact)\)|\bevaluation data\b"
    r"|^(supplementary material|data(set)? for)\b"
    r"|^[\w.-]+/[\w.-]+:\s",  # GitHub releases on Zenodo: "owner/repo: v1.2"
    re.IGNORECASE,
)


def is_research_paper(pub):
    """Remove artifacts and data packages; they repeat a paper and add noise"""
    return not ARTIFACT_TITLE.search(pub.get("title") or "")


def keep_only_confirmed(by_source, identity):
    """Drop the papers of name-only matches that no better-identified source confirms.

    A name-only match can be a namesake. When another source identified the
    member by ORCID iD, profile link, affiliation, or a unique name, the tool
    keeps a name-only paper only if that source has the same title. The kept
    papers still add abstracts and PDF links.
    """
    strong = [src for src, method in identity.items() if method and method != "name-only"]
    weak = [src for src, method in identity.items() if method == "name-only"]
    if not strong or not weak:
        return [pub for pubs in by_source.values() for pub in pubs]
    confirmed_titles = {normalize_title(pub.get("title")) for src in strong for pub in by_source[src]}
    publications = []
    for src, pubs in by_source.items():
        if src in weak:
            pubs = [pub for pub in pubs if normalize_title(pub.get("title")) in confirmed_titles]
        publications.extend(pubs)
    return publications


def collect_member_publications(member_info, sources, dblp_index=None):
    """Return (publications, identity) for one PC member from all chosen sources"""
    by_source = {}
    identity = {}

    if "dblp" in sources:
        by_source["dblp"], identity["dblp"] = fetch_publications_from_dblp(member_info, dblp_index)

    if "openalex" in sources:
        by_source["openalex"], identity["openalex"] = fetch_publications_from_openalex(member_info)

    if "semanticscholar" in sources:
        by_source["semanticscholar"], identity["semanticscholar"] = (
            fetch_publications_from_semantic_scholar(member_info)
        )

    publications = keep_only_confirmed(by_source, identity)
    publications = dedupe_publications([p for p in publications if is_research_paper(p)])

    if "website" in sources:
        website_data = fetch_publications_from_website(member_info)
        website_publications = list(website_data.get("publications", []))
        for bibtex_file in website_data.get("bibtex", []):
            bibtex_content = download_bibtex_content(bibtex_file["url"])
            if bibtex_content:
                website_publications.extend(parse_bibtex_publications(bibtex_content))
        publications = dedupe_publications(
            attach_website_publications(publications, website_publications, website_data.get("pdfs", []))
        )

    return publications, {k: v for k, v in identity.items() if v}


def build_reviewer_profiles(pc_list, sources, min_pdf, text_source="auto", dblp_index=None):
    profiles = {}
    summary = {
        "existing_by_source": {},
        "recovered_by_source": {},
        "failures_by_source": {},
        "total_publications": 0,
        "total_with_pdf": 0,
    }

    def collect(member_info):
        publications, identity = collect_member_publications(member_info, sources, dblp_index)
        member_info["identity"] = identity
        stats = enrich_pdf_urls(publications, text_source)
        return member_info, publications, identity, stats

    for member_info, publications, identity, stats in run_parallel(
        collect, pc_list, "Collecting publications", "member"
    ):
        name = member_info["name"]
        for key in ["existing_by_source", "recovered_by_source", "failures_by_source"]:
            for src, count in stats[key].items():
                summary[key][src] = summary[key].get(src, 0) + count
        summary["total_publications"] += stats["total_publications"]
        summary["total_with_pdf"] += stats["total_with_pdf"]

        if not publications:
            print(f"No publications found for {name}")
            continue

        pdf_count = sum(1 for pub in publications if pub.get("pdf_url"))
        if min_pdf and pdf_count < min_pdf:
            print(f"Skipping {name}: {pdf_count} PDFs < min {min_pdf}")
            continue

        # Two members can share a display name; keep both
        key = name if name not in profiles else f"{name} ({member_info.get('affiliation') or len(profiles)})"
        profiles[key] = {
            "publications": publications,
            "info": member_info,
            "identity": identity,
            "pdf_summary": stats,
        }

    print(f"Built {len(profiles)} reviewer profiles")
    return profiles, summary

# ----------- Step 5: Parse Submission PDF ----------- #

def extract_text_from_pdf(pdf_path):
    """Return the text of the submission.

    pdfminer's default layout analysis keeps the two columns apart and keeps
    the spaces between words. markitdown mixed figure code into the abstract
    and glued words together, which broke the embedding scorer.
    """
    from pdfminer.high_level import extract_text

    try:
        with suppress_pdf_warnings():
            return extract_text(pdf_path, maxpages=PDF_MAX_PAGES) or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# ----------- Step 6: Conflicts of Interest ----------- #

def detect_conflicts(member_info, publications, submission_authors, submission_affiliations, coi_years):
    """Return the reasons why a PC member has a conflict with the submission"""
    conflicts = []
    name = member_info.get("name", "")
    for author in submission_authors:
        if names_match(name, author):
            conflicts.append(f"is an author of the submission ({author})")

    min_year = CURRENT_YEAR - coi_years
    for pub in publications:
        if not pub.get("year") or pub["year"] < min_year:
            continue
        for coauthor in pub.get("authors") or []:
            if not coauthor or names_match(name, coauthor):
                continue
            for author in submission_authors:
                if names_match(coauthor, author):
                    conflicts.append(f"co-author of {author}: \"{pub.get('title')}\" ({pub['year']})")

    for member_aff in member_affiliations(member_info):
        for affiliation in submission_affiliations:
            if affiliation_similarity(member_aff, affiliation) >= 0.8:
                conflicts.append(f"same affiliation as the authors ({member_aff})")
    return list(dict.fromkeys(conflicts))

# ----------- Step 7: Similarity Scoring ----------- #



EMBEDDING_CACHE_PATH = HTML_CACHE_DIR.parent / "embeddings.sqlite"


def cached_embeddings(documents, embedding_model, model_class):
    """Encode documents, reusing the vectors of documents that an earlier run encoded"""
    import sqlite3

    keys = [hashlib.sha256(f"{embedding_model}\n{doc}".encode("utf-8")).hexdigest() for doc in documents]
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EMBEDDING_CACHE_PATH))
    conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
    found = {}
    unique_keys = list(dict.fromkeys(keys))
    for start in range(0, len(unique_keys), 900):
        chunk = unique_keys[start : start + 900]
        rows = conn.execute(
            f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(chunk))})", chunk
        ).fetchall()
        found.update({key: np.frombuffer(vec, dtype=np.float32) for key, vec in rows})

    missing = {key: doc for key, doc in zip(keys, documents) if key not in found}
    print(f"Embeddings: {len(found)} from cache, {len(missing)} to encode")
    if missing:
        model = model_class(embedding_model)
        vectors = model.encode(
            list(missing.values()), batch_size=32, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        rows = [(key, vec.tobytes()) for key, vec in zip(missing, vectors)]
        conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)
        conn.commit()
        found.update(zip(missing, vectors))
    conn.close()
    return np.vstack([found[key] for key in keys])


def vectorize_documents(documents, scorer, embedding_model=SPECTER_MODEL):
    """Return one L2-normalized row per document"""
    if scorer == "embedding":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "The embedding scorer needs sentence-transformers. "
                "Install it with: pip install -e '.[embeddings]'"
            )
        return cached_embeddings(documents, embedding_model, SentenceTransformer)

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectors = vectorizer.fit_transform(documents)
    if scorer == "lsa":
        n_components = min(LSA_COMPONENTS, vectors.shape[0] - 1, vectors.shape[1] - 1)
        if n_components >= 2:
            vectors = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(vectors)
            return normalize(vectors)
    return normalize(vectors)


def as_dense_row(matrix_row):
    return np.asarray(matrix_row.toarray() if hasattr(matrix_row, "toarray") else matrix_row).ravel()


def compute_similarity(
    profiles,
    paper_text,
    text_source,
    top_papers=3,
    scorer="tfidf",
    top_k=DEFAULT_TOP_K,
    alpha=DEFAULT_ALPHA,
    half_life=DECAY_HALF_LIFE,
    embedding_model=SPECTER_MODEL,
):
    """Rank reviewers by a mix of their closest papers and their whole profile.

    score = alpha * mean of the top_k decayed paper similarities (missing papers count as 0)
          + (1 - alpha) * similarity of the submission to the decayed mean profile vector
    """
    all_publications = []
    for profile_data in profiles.values():
        all_publications.extend(profile_data["publications"])
    prefetch_pdf_texts(all_publications, text_source)

    documents = []
    paper_metadata = []
    for reviewer, profile_data in profiles.items():
        for pub in profile_data["publications"]:
            text = normalize_text(publication_document(pub, text_source))
            if not text:
                continue
            documents.append(text)
            paper_metadata.append((reviewer, pub))

    if not documents:
        return []

    cleaned = clean_paper_text(paper_text)
    # SPECTER is trained on title + abstract; TF-IDF and LSA use the whole cleaned text
    submission_doc = extract_abstract(paper_text, limit=DOC_WORD_LIMIT) if scorer == "embedding" else cleaned
    vectors = vectorize_documents(documents + [submission_doc or cleaned], scorer, embedding_model)
    paper_vec = as_dense_row(vectors[-1])
    pub_vecs = vectors[:-1]
    scores = np.asarray(pub_vecs @ paper_vec).ravel()

    rows_by_reviewer = {}
    for row, ((reviewer, pub), score) in enumerate(zip(paper_metadata, scores)):
        weight = exponential_decay_weight(pub.get("year"), half_life=half_life)
        rows_by_reviewer.setdefault(reviewer, []).append(
            {"paper": pub, "score": float(score), "weight": weight, "weighted_score": float(score) * weight, "row": row}
        )

    ranked_reviewers = []
    for reviewer, papers in rows_by_reviewer.items():
        papers.sort(key=lambda x: x["weighted_score"], reverse=True)
        top_k_score = sum(p["weighted_score"] for p in papers[:top_k]) / top_k

        weights = np.array([p["weight"] for p in papers])
        rows = [p["row"] for p in papers]
        centroid = np.asarray(pub_vecs[rows].T @ weights).ravel()
        norm = np.linalg.norm(centroid)
        profile_score = float(centroid @ paper_vec / norm) if norm else 0.0

        profile = profiles[reviewer]
        ranked_reviewers.append(
            {
                "name": reviewer,
                "score": alpha * top_k_score + (1 - alpha) * profile_score,
                "top_k_score": top_k_score,
                "profile_score": profile_score,
                "papers": [
                    {"paper": p["paper"], "score": p["score"], "weighted_score": p["weighted_score"]}
                    for p in papers[:top_papers]
                ],
                "identity": profile.get("identity", {}),
                "conflicts": profile.get("conflicts", []),
                "pdf_summary": profile.get("pdf_summary", {}),
            }
        )

    ranked_reviewers.sort(key=lambda x: x["score"], reverse=True)
    return ranked_reviewers

# ----------- Output Rendering ----------- #

def identity_text(identity):
    if not identity:
        return "not found in any source"
    return ", ".join(f"{src}: {method}" for src, method in sorted(identity.items()))


def render_reviewer_lines(ranked_reviewers, top_n, markdown):
    lines = []
    bullet = "   - " if markdown else "   "
    for idx, reviewer in enumerate(ranked_reviewers[:top_n], 1):
        name = f"**{reviewer['name']}**" if markdown else reviewer["name"]
        lines.append(
            f"{idx}. {name} - Score: {reviewer['score']:.4f} "
            f"(top-k {reviewer['top_k_score']:.4f}, profile {reviewer['profile_score']:.4f})"
        )
        lines.append(f"{bullet}Identity: {identity_text(reviewer.get('identity'))}")
        for conflict in reviewer.get("conflicts", []):
            lines.append(f"{bullet}CONFLICT: {conflict}")
        pdf_summary = reviewer.get("pdf_summary", {})
        if pdf_summary:
            lines.append(
                f"{bullet}PDFs: {pdf_summary.get('total_with_pdf', 0)}/{pdf_summary.get('total_publications', 0)}"
            )
        for paper in reviewer["papers"]:
            pub = paper["paper"]
            title = pub.get("title") or "Untitled"
            year = pub.get("year") or ""
            venue = pub.get("venue") or ""
            suffix = f" ({year})" if year else ""
            venue_text = f" {venue}" if venue else ""
            lines.append(
                f"   - {title}{suffix}{venue_text} "
                f"[score {paper['score']:.4f}, after decay {paper['weighted_score']:.4f}]"
            )
    return "\n".join(lines)


def render_table(ranked_reviewers, top_n):
    return render_reviewer_lines(ranked_reviewers, top_n, markdown=False)


def render_markdown(ranked_reviewers, top_n):
    return render_reviewer_lines(ranked_reviewers, top_n, markdown=True)


def render_json(ranked_reviewers, top_n):
    return json.dumps({"reviewers": ranked_reviewers[:top_n]}, indent=2, default=str)

# ----------- Main Function ----------- #

def open_dblp_index(dblp_index_path):
    path = Path(dblp_index_path) if dblp_index_path else DEFAULT_INDEX_PATH
    if path.exists():
        print(f"Using local DBLP index: {path}")
        return DblpIndex(path)
    if dblp_index_path:
        raise FileNotFoundError(f"DBLP index not found: {path}")
    return None


def match_reviewers(
    cfp_url,
    submission_pdf,
    sources,
    text_source,
    output_format,
    top_n=8,
    top_papers=3,
    min_pdf=0,
    hotcrp_url=None,
    scorer="tfidf",
    top_k=DEFAULT_TOP_K,
    alpha=DEFAULT_ALPHA,
    half_life=DECAY_HALF_LIFE,
    embedding_model=SPECTER_MODEL,
    dblp_index_path=None,
    submission_authors=None,
    submission_affiliations=None,
    coi_years=DEFAULT_COI_YEARS,
    coi_mode="flag",
    max_members=None,
    profile_mode="lazy",
    profile_top=DEFAULT_PROFILE_TOP,
):
    print("Parsing submission PDF...")
    paper_text = extract_text_from_pdf(submission_pdf)
    if not paper_text.strip():
        raise ValueError(
            "Failed to extract text from submission PDF. "
            "Install pdfminer.six or use a different PDF."
        )

    print("Extracting PC members...")
    pc_list = extract_pc_members(
        cfp_url, hotcrp_url=hotcrp_url, max_members=max_members,
        fetch_profiles=(profile_mode == "all"),
    )
    print(f"Found {len(pc_list)} PC members" + (" (limited by --max-members)" if max_members else ""))

    dblp_index = open_dblp_index(dblp_index_path) if "dblp" in sources else None

    print("Building reviewer profiles...")
    profiles, _ = build_reviewer_profiles(
        pc_list, sources, min_pdf, text_source=text_source, dblp_index=dblp_index
    )

    def refresh(members, reason):
        """Fetch the researchr profiles of members, then collect their publications again"""
        members = [m for m in members if m.get("url") and not m.get("profile_fetched")]
        if not members:
            return False
        print(f"Fetching researchr profiles for {len(members)} members ({reason})")
        add_profile_details(members)
        members = [m for m in members if m.get("profile_fetched")]
        if not members:
            return False
        refreshed, _ = build_reviewer_profiles(
            members, sources, min_pdf, text_source=text_source, dblp_index=dblp_index
        )
        for key in [k for k, v in profiles.items() if any(v["info"] is m for m in members)]:
            del profiles[key]
        profiles.update(refreshed)
        return True

    if profile_mode == "lazy":
        ambiguous = [
            m for m in pc_list
            if not CONFIDENT_IDENTITY_METHODS & set((m.get("identity") or {}).values())
        ]
        refresh(ambiguous, "no source identified them by ORCID iD or affiliation")

    submission_authors = submission_authors or []
    submission_affiliations = submission_affiliations or []

    def apply_conflicts():
        if not (submission_authors or submission_affiliations):
            return
        for reviewer in list(profiles):
            profile = profiles[reviewer]
            conflicts = detect_conflicts(
                profile["info"], profile["publications"], submission_authors,
                submission_affiliations, coi_years,
            )
            profile["conflicts"] = conflicts
            if conflicts and coi_mode == "exclude":
                print(f"Excluding {reviewer}: {conflicts[0]}")
                del profiles[reviewer]


    def score():
        apply_conflicts()
        print(f"Computing similarity scores with the {scorer} scorer...")
        return compute_similarity(
            profiles,
            paper_text,
            text_source,
            top_papers=top_papers,
            scorer=scorer,
            top_k=top_k,
            alpha=alpha,
            half_life=half_life,
            embedding_model=embedding_model,
        )

    ranked_reviewers = score()

    if profile_mode == "lazy":
        # The website of a top reviewer can add PDF links and abstracts; rank again if it does
        top_members = [profiles[r["name"]]["info"] for r in ranked_reviewers[:profile_top]]
        if refresh(top_members, f"top {profile_top} reviewers"):
            ranked_reviewers = score()

    summary = {
        "existing_by_source": {},
        "recovered_by_source": {},
        "failures_by_source": {},
        "total_publications": 0,
        "total_with_pdf": 0,
    }
    for profile in profiles.values():
        stats = profile.get("pdf_summary", {})
        for key in ["existing_by_source", "recovered_by_source", "failures_by_source"]:
            for src, count in stats.get(key, {}).items():
                summary[key][src] = summary[key].get(src, 0) + count
        summary["total_publications"] += stats.get("total_publications", 0)
        summary["total_with_pdf"] += stats.get("total_with_pdf", 0)

    if output_format == "json":
        output = render_json(ranked_reviewers, top_n)
    elif output_format == "markdown":
        output = render_markdown(ranked_reviewers, top_n)
    else:
        output = render_table(ranked_reviewers, top_n)

    print(output)
    if SOURCE_FAILURES:
        failed = ", ".join(f"{src}: {count}" for src, count in sorted(SOURCE_FAILURES.items()))
        print(f"\nMembers with a failed source (results for them are less complete): {failed}")
    print("\nPDF recovery summary:")
    print(f"- Publications scanned: {summary['total_publications']}")
    print(f"- Publications with PDFs: {summary['total_with_pdf']}")
    if summary["existing_by_source"]:
        existing = ", ".join(
            f"{src}: {count}" for src, count in sorted(summary["existing_by_source"].items())
        )
        print(f"- Existing PDFs by source: {existing}")
    if summary["recovered_by_source"]:
        recovered = ", ".join(
            f"{src}: {count}" for src, count in sorted(summary["recovered_by_source"].items())
        )
        print(f"- Recovered PDFs by source: {recovered}")
    if summary["failures_by_source"]:
        failures = ", ".join(
            f"{src}: {count}" for src, count in sorted(summary["failures_by_source"].items())
        )
        print(f"- Recovery failures by source: {failures}")
    return ranked_reviewers

# Example usage:
# match_reviewers(
#     "https://conf.researchr.org/track/ase-2025/ase-2025-papers",
#     "./example_submission.pdf",
#     sources=["dblp", "openalex", "semanticscholar", "website"],
#     text_source="auto",
#     output_format="table",
#     min_pdf=0,
# )
