# tpms_matcher.py

import json
import math
import os
import re
import tempfile
import time
import io
import logging
from contextlib import contextmanager, redirect_stderr
from datetime import datetime
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Config ----------- #
DBLP_SEARCH_API = "https://dblp.org/search/publ/api?q=author:{}&format=json&h={}"
OPENALEX_AUTHORS_API = "https://api.openalex.org/authors"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_AUTHOR_SEARCH = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR = "https://api.semanticscholar.org/graph/v1/author/{}"
CROSSREF_WORKS_API = "https://api.crossref.org/works/{}"
DECAY_HALF_LIFE = 5  # years
MARKITDOWN = MarkItDown()
CURRENT_YEAR = datetime.now().year
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": "tpms_matcher/0.1 (+https://github.com/imranur-rahman/tpms_matcher)"
}
MAX_SEMANTIC_SCHOLAR_PAPERS = 200

PDF_TEXT_CACHE = {}

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
    age = current_year - int(year)
    return math.exp(-math.log(2) * age / half_life)


def fetch_html(url):
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def fetch_json(url, params=None):
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, params=params)
    resp.raise_for_status()
    return resp.json()


def normalize_title(title):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def merge_publications(existing, incoming):
    for key in ["title", "year", "venue", "url", "pdf_url", "abstract", "authors", "doi", "arxiv_id"]:
        if not existing.get(key) and incoming.get(key):
            existing[key] = incoming[key]
    if incoming.get("abstract") and (
        not existing.get("abstract") or len(incoming["abstract"]) > len(existing["abstract"])
    ):
        existing["abstract"] = incoming["abstract"]
    if incoming.get("pdf_url") and not existing.get("pdf_url"):
        existing["pdf_url"] = incoming["pdf_url"]
    existing_sources = set(existing.get("sources", []))
    incoming_sources = set(incoming.get("sources", []))
    existing["sources"] = sorted(existing_sources | incoming_sources)
    return existing


def dedupe_publications(publications):
    deduped = {}
    for pub in publications:
        title = pub.get("title") or ""
        year = pub.get("year") or ""
        key = (normalize_title(title), str(year))
        if key in deduped:
            deduped[key] = merge_publications(deduped[key], pub)
        else:
            pub.setdefault("sources", [])
            deduped[key] = pub
    return list(deduped.values())


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


def download_pdf_text(pdf_url):
    if pdf_url in PDF_TEXT_CACHE:
        return PDF_TEXT_CACHE[pdf_url]

    try:
        resp = requests.get(pdf_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmpf:
            tmpf.write(resp.content)
            tmpf.flush()
            with suppress_pdf_warnings():
                result = MARKITDOWN.convert(tmpf.name)
            text = result.text_content or ""
    except Exception:
        text = ""

    PDF_TEXT_CACHE[pdf_url] = text
    return text


def select_publication_text(pub, text_source):
    if text_source == "title":
        return pub.get("title", "")
    if text_source == "abstract":
        return pub.get("abstract") or pub.get("title", "")
    if text_source == "pdf":
        pdf_url = pub.get("pdf_url")
        return download_pdf_text(pdf_url) if pdf_url else ""

    # auto fallback: pdf -> abstract -> title
    pdf_url = pub.get("pdf_url")
    if pdf_url:
        pdf_text = download_pdf_text(pdf_url)
        if pdf_text:
            return pdf_text
    return pub.get("abstract") or pub.get("title", "")


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def prefetch_pdf_texts(publications, text_source):
    if text_source not in {"auto", "pdf"}:
        return
    pdf_pubs = [pub for pub in publications if pub.get("pdf_url")]
    total_pubs = len(publications)
    total_pdfs = len(pdf_pubs)
    print(f"PDF parsing plan: {total_pdfs}/{total_pubs} publications have PDFs")
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(pdf_pubs, desc="Parsing PDFs", unit="pdf")
    except Exception:
        iterator = pdf_pubs
    for pub in iterator:
        download_pdf_text(pub.get("pdf_url"))

def normalize_doi(doi):
    if not doi:
        return ""
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
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

def enrich_pdf_urls(publications):
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
        print(f"Fetching profile details from {profile_url}")
        html = fetch_html(profile_url)
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
                profile_details["orcid"] = content
            elif heading == "dblp":
                profile_details["dblp"] = content

        return profile_details

    except Exception as e:
        print(f"Error fetching profile details from {profile_url}: {e}")
        return {}


def extract_pc_members(cfp_url):
    html = fetch_html(cfp_url)
    soup = BeautifulSoup(html, "html.parser")

    # Extract detailed member information
    pc_members = []

    # Look for list items with class "list-group-item" (specific structure)
    for li in soup.find_all("li", class_="list-group-item"):
        member_info = {}

        # Extract profile URL
        link = li.find("a", class_="navigate")
        if link:
            member_info["url"] = link.get("href")
            if member_info["url"] and not member_info["url"].startswith("http"):
                member_info["url"] = urljoin(cfp_url, member_info["url"])
        else:
            member_info["url"] = None

        # Extract name and role from first media-heading
        name_headings = li.find_all("h5", class_="media-heading")
        if len(name_headings) >= 1:
            # First heading contains the name and possibly a role
            name_heading = name_headings[0]

            # Extract role from pull-right span if it exists
            role_span = name_heading.find("span", class_="pull-right")
            if role_span:
                role_text = role_span.get_text().strip()
                member_info["role"] = role_text if role_text else None
                # Remove the role span to get clean name
                role_span.extract()
            else:
                member_info["role"] = None

            # Now get the clean name
            original_name = name_heading.get_text().strip()
            member_info["name"] = original_name
            member_info["decoded_name"] = original_name.encode().decode("unicode_escape")

            # Second heading contains university and possibly location
            if len(name_headings) >= 2:
                affiliation_text = name_headings[1].get_text().strip()

                # Parse affiliation - could be "University" or "University, Country"
                if "," in affiliation_text:
                    parts = [part.strip() for part in affiliation_text.split(",")]
                    member_info["university"] = parts[0]
                    member_info["location"] = ", ".join(parts[1:])
                else:
                    member_info["university"] = affiliation_text
                    member_info["location"] = None
            else:
                member_info["university"] = None
                member_info["location"] = None

            # Extract detailed profile information if URL is available
            if member_info["url"]:
                profile_details = extract_profile_details(member_info["url"])
                member_info.update(profile_details)
                # Add a small delay to be respectful
                time.sleep(0.5)

            # Only add if we have a valid name
            if member_info["name"] and len(member_info["name"].split()) >= 2:
                pc_members.append(member_info)

    # Fallback to original extraction method if the specific structure isn't found
    if not pc_members:
        print("No specific list-group-item structure found, falling back to generic extraction.")
        for tag in soup.find_all(["li", "p", "td"]):
            text = tag.get_text().strip()
            if len(text.split()) >= 2 and not text.lower().startswith("program"):
                # Try to extract URL if it's within a link
                link = tag.find("a")
                url = None
                if link:
                    url = link.get("href")
                    if url and not url.startswith("http"):
                        url = urljoin(cfp_url, url)

                member_info = {
                    "name": text,
                    "decoded_name": text.encode().decode("unicode_escape"),
                    "url": url,
                    "university": None,
                    "location": None,
                    "role": None,
                }

                # Extract detailed profile information if URL is available
                if url:
                    profile_details = extract_profile_details(url)
                    member_info.update(profile_details)
                    time.sleep(0.5)

                pc_members.append(member_info)

    return pc_members

# ----------- Step 2: External Publication Sources ----------- #

def fetch_publications_from_dblp(author_name, max_hits=200):
    publications = []
    try:
        encoded = quote(author_name)
        url = DBLP_SEARCH_API.format(encoded, max_hits)
        data = fetch_json(url)
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        for hit in hits:
            info = hit.get("info", {})
            title = info.get("title")
            year = info.get("year")
            venue = info.get("venue") or info.get("booktitle")
            url = info.get("url")
            doi = None
            ee = info.get("ee")
            if isinstance(ee, list):
                ee = ee[0] if ee else None
            if isinstance(ee, str) and "doi.org/" in ee:
                doi = ee.split("doi.org/", 1)[1]
            authors_data = info.get("authors", {}).get("author", [])
            if isinstance(authors_data, dict):
                authors = [authors_data.get("text")]
            else:
                authors = [a.get("text") for a in authors_data if isinstance(a, dict)]

            if isinstance(year, list) and year:
                year = year[0]
            if title and year and str(year).isdigit():
                publications.append(
                    {
                        "title": title,
                        "year": int(year),
                        "venue": venue,
                        "url": url,
                        "authors": authors,
                        "abstract": "",
                        "pdf_url": "",
                        "doi": doi,
                        "arxiv_id": "",
                        "sources": ["dblp"],
                    }
                )
    except Exception as e:
        print(f"Error fetching DBLP publications for {author_name}: {e}")

    return publications


def fetch_publications_from_openalex(author_name, per_page=200):
    publications = []
    try:
        clean_name = re.sub(r"\s+", " ", author_name).strip()
        params = {"search": clean_name, "per_page": 1}
        openalex_email = os.environ.get("OPENALEX_EMAIL")
        if openalex_email:
            params["mailto"] = openalex_email
        author_data = fetch_json(OPENALEX_AUTHORS_API, params=params)
        author_results = author_data.get("results", [])
        if not author_results:
            return publications
        author_id = author_results[0].get("id")
        if not author_id:
            return publications
        works_params = {"filter": f"author.id:{author_id}", "per_page": per_page}
        if openalex_email:
            works_params["mailto"] = openalex_email
        data = fetch_json(OPENALEX_WORKS_API, params=works_params)
        for work in data.get("results", []):
            title = work.get("title")
            year = work.get("publication_year")
            venue = work.get("host_venue", {}).get("display_name")
            url = work.get("id")
            doi = None
            arxiv_id = None
            ids = work.get("ids", {})
            if isinstance(ids, dict):
                doi = ids.get("doi")
                arxiv_id = ids.get("arxiv")
            abstract = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
            authors = [
                auth.get("author", {}).get("display_name")
                for auth in work.get("authorships", [])
            ]
            best_oa = work.get("best_oa_location") or {}
            pdf_url = best_oa.get("pdf_url") or ""

            if title and year:
                publications.append(
                    {
                        "title": title,
                        "year": int(year),
                        "venue": venue,
                        "url": url,
                        "authors": authors,
                        "abstract": abstract,
                        "pdf_url": pdf_url,
                        "doi": doi,
                        "arxiv_id": arxiv_id,
                        "sources": ["openalex"],
                    }
                )
    except Exception as e:
        print(f"Error fetching OpenAlex publications for {author_name}: {e}")

    return publications

def fetch_publications_from_semantic_scholar(author_name, limit=1):
    publications = []
    try:
        params = {
            "query": author_name,
            "limit": limit,
            "fields": "authorId,name,paperCount",
        }
        data = fetch_json(SEMANTIC_SCHOLAR_AUTHOR_SEARCH, params=params)
        results = data.get("data", [])
        if not results:
            return publications
        author_id = results[0].get("authorId")
        if not author_id:
            return publications
        fields = (
            "papers.title,papers.year,papers.venue,papers.url,papers.authors,"
            "papers.openAccessPdf,papers.externalIds"
        )
        offset = 0
        while offset < MAX_SEMANTIC_SCHOLAR_PAPERS:
            author_data = fetch_json(
                SEMANTIC_SCHOLAR_AUTHOR.format(author_id),
                params={"fields": fields, "limit": 100, "offset": offset},
            )
            papers = author_data.get("papers", [])
            if not papers:
                break
            for paper in papers:
                title = paper.get("title")
                year = paper.get("year")
                venue = paper.get("venue")
                url = paper.get("url")
                authors = [a.get("name") for a in paper.get("authors", []) if a.get("name")]
                open_access = paper.get("openAccessPdf") or {}
                pdf_url = open_access.get("url") or ""
                external_ids = paper.get("externalIds") or {}
                doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
                arxiv_id = external_ids.get("ArXiv") if isinstance(external_ids, dict) else None
                if title and year:
                    publications.append(
                        {
                            "title": title,
                            "year": int(year),
                            "venue": venue,
                            "url": url,
                            "authors": authors,
                            "abstract": "",
                            "pdf_url": pdf_url,
                            "doi": doi,
                            "arxiv_id": arxiv_id,
                            "sources": ["semanticscholar"],
                        }
                    )
            offset += len(papers)
    except Exception as e:
        print(f"Error fetching Semantic Scholar publications for {author_name}: {e}")

    return publications

# ----------- Step 3: Parse Reviewers Website ----------- #

def fetch_publications_from_website(member_info):
    """Crawl personal website to extract PDF and BibTeX files"""
    personal_website = member_info.get("personal_website")
    if not personal_website:
        return {"pdfs": [], "bibtex": [], "publications": []}

    print(f"Crawling personal website for {member_info['name']}: {personal_website}")

    try:
        html = fetch_html(personal_website)
        soup = BeautifulSoup(html, "html.parser")

        # Extract all links from the website
        links = soup.find_all("a", href=True)

        pdfs = []
        bibtex_files = []
        publications = []

        for link in links:
            href = link.get("href")
            if not href:
                continue

            # Convert relative URLs to absolute
            if href.startswith("/") or not href.startswith("http"):
                full_url = urljoin(personal_website, href)
            else:
                full_url = href

            # Extract PDF files
            link_text = link.get_text().strip()
            if is_probable_pdf_link(href, link_text):
                pdf_info = {
                    "url": full_url,
                    "title": link_text or "Unknown Title",
                    "filename": href.split("/")[-1],
                }
                pdfs.append(pdf_info)

            # Extract BibTeX files
            elif href.lower().endswith(".bib") or "bibtex" in href.lower():
                bibtex_info = {
                    "url": full_url,
                    "title": link.get_text().strip() or "BibTeX File",
                    "filename": href.split("/")[-1],
                }
                bibtex_files.append(bibtex_info)

        # Look for publication sections in the HTML
        publications = extract_publications_from_html(soup, personal_website)

        # Try to find common publication page patterns
        pub_patterns = ["publications", "papers", "research", "pubs"]
        for pattern in pub_patterns:
            pub_links = [
                link
                for link in links
                if pattern in link.get("href", "").lower()
                or pattern in link.get_text().lower()
            ]
            for pub_link in pub_links[:3]:  # Limit to first 3 matches
                pub_url = urljoin(personal_website, pub_link.get("href"))
                try:
                    sub_publications = crawl_publication_page(pub_url)
                    publications.extend(sub_publications)
                    time.sleep(0.5)  # Be respectful
                except Exception as e:
                    print(f"Error crawling publication page {pub_url}: {e}")
                    continue

        result = {"pdfs": pdfs, "bibtex": bibtex_files, "publications": publications}

        print(
            f"Found {len(pdfs)} PDFs, {len(bibtex_files)} BibTeX files, {len(publications)} publications"
        )
        return result

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

        if title:
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
    """Extract publications from unstructured text"""
    publications = []

    # Look for text that might contain publication lists
    text_elements = soup.find_all(["p", "div", "li"])

    for element in text_elements:
        text = element.get_text()

        # Look for patterns that suggest this is a publication
        if re.search(r"\b(19|20)\d{2}\b", text) and len(text.split()) > 5:
            # This might be a publication entry
            title_match = re.match(r"^([^.]+\.)", text)
            year_match = re.search(r"\b(19|20)\d{2}\b", text)

            if title_match and year_match:
                title = title_match.group(1).strip(".")
                year = int(year_match.group())

                # Look for PDF links in this element
                pdf_url = None
                bibtex_url = None
                links = element.find_all("a", href=True)
                for link in links:
                    href = link.get("href")
                    link_text = link.get_text().strip()
                    if is_probable_pdf_link(href, link_text):
                        pdf_url = urljoin(base_url, href)
                    elif href.lower().endswith(".bib"):
                        bibtex_url = urljoin(base_url, href)

                publications.append(
                    {
                        "title": title,
                        "year": year,
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


def crawl_publication_page(pub_url):
    """Crawl a dedicated publications page"""
    publications = []

    try:
        html = fetch_html(pub_url)
        soup = BeautifulSoup(html, "html.parser")
        publications = extract_publications_from_html(soup, pub_url)
    except Exception as e:
        print(f"Error crawling publication page {pub_url}: {e}")

    return publications


def download_bibtex_content(bibtex_url):
    """Download and parse BibTeX content"""
    try:
        response = requests.get(bibtex_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error downloading BibTeX from {bibtex_url}: {e}")
        return None


def parse_bibtex_publications(bibtex_content):
    """Parse BibTeX content to extract publication information"""
    publications = []

    # Simple BibTeX parsing (you might want to use a proper BibTeX library)
    entries = re.findall(r"@\w+\{[^}]+\}", bibtex_content, re.DOTALL)

    for entry in entries:
        title_match = re.search(r"title\s*=\s*[\"{]([^\"}]+)[\"}]", entry, re.IGNORECASE)
        year_match = re.search(r"year\s*=\s*[\"{]?(\d{4})[\"}]?", entry, re.IGNORECASE)

        if title_match:
            title = title_match.group(1)
            year = int(year_match.group(1)) if year_match else None

            publications.append(
                {
                    "title": title,
                    "year": year,
                    "venue": None,
                    "abstract": "",
                    "url": "",
                    "authors": [],
                    "pdf_url": "",
                    "doi": "",
                    "arxiv_id": "",
                    "sources": ["bibtex"],
                }
            )

    return publications

# ----------- Step 4: Build Reviewer Profiles ----------- #

def build_reviewer_profiles(pc_list, sources, min_pdf):
    profiles = {}
    summary = {
        "existing_by_source": {},
        "recovered_by_source": {},
        "failures_by_source": {},
        "total_publications": 0,
        "total_with_pdf": 0,
    }
    for member_info in pc_list:
        name = member_info["name"]

        publications = []
        website_data = {"pdfs": [], "bibtex": [], "publications": []}

        if "website" in sources:
            website_data = fetch_publications_from_website(member_info)
            website_publications = website_data.get("publications", [])

            # Download and parse BibTeX files
            for bibtex_file in website_data.get("bibtex", []):
                bibtex_content = download_bibtex_content(bibtex_file["url"])
                if bibtex_content:
                    bibtex_pubs = parse_bibtex_publications(bibtex_content)
                    website_publications.extend(bibtex_pubs)

            publications.extend(website_publications)

        if "dblp" in sources:
            publications.extend(fetch_publications_from_dblp(name))

        if "openalex" in sources:
            publications.extend(fetch_publications_from_openalex(name))

        if "semanticscholar" in sources:
            publications.extend(fetch_publications_from_semantic_scholar(name))

        publications = dedupe_publications(publications)
        stats = enrich_pdf_urls(publications)
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

        # Store additional website data
        member_info["website_data"] = website_data

        profiles[name] = {
            "publications": publications,
            "info": member_info,
            "pdf_summary": stats,
        }

        print(f"Built profile for {name}: {len(publications)} publications")

    return profiles, summary

# ----------- Step 5: Parse Submission PDF ----------- #

def extract_text_from_pdf(pdf_path):
    try:
        with suppress_pdf_warnings():
            result = MARKITDOWN.convert(pdf_path)
        return result.text_content or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# ----------- Step 6: Similarity Scoring ----------- #

def compute_similarity(profiles, paper_text, text_source, top_papers=3):
    corpus = []
    paper_metadata = []

    all_publications = []
    for profile_data in profiles.values():
        all_publications.extend(profile_data["publications"])
    prefetch_pdf_texts(all_publications, text_source)

    for reviewer, profile_data in profiles.items():
        for pub in profile_data["publications"]:
            text = select_publication_text(pub, text_source)
            if not text:
                continue
            corpus.append(normalize_text(text))
            paper_metadata.append((reviewer, pub))

    corpus.append(normalize_text(paper_text))

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(corpus)
    paper_vec = vectors[-1]
    pub_vecs = vectors[:-1]

    scores = cosine_similarity(paper_vec, pub_vecs)[0]
    reviewer_papers = {}

    for (reviewer, pub), score in zip(paper_metadata, scores):
        reviewer_papers.setdefault(reviewer, []).append({"paper": pub, "score": float(score)})

    ranked_reviewers = []
    for reviewer, papers in reviewer_papers.items():
        papers.sort(key=lambda x: x["score"], reverse=True)
        top_scores = [p["score"] for p in papers[: min(3, len(papers))]]
        aggregate_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        ranked_reviewers.append(
            {
                "name": reviewer,
                "score": aggregate_score,
                "papers": papers[:top_papers],
                "pdf_summary": profiles[reviewer].get("pdf_summary", {}),
            }
        )

    ranked_reviewers.sort(key=lambda x: x["score"], reverse=True)
    return ranked_reviewers

# ----------- Output Rendering ----------- #

def render_table(ranked_reviewers, top_n):
    lines = []
    for idx, reviewer in enumerate(ranked_reviewers[:top_n], 1):
        lines.append(f"{idx}. {reviewer['name']} - Score: {reviewer['score']:.4f}")
        pdf_summary = reviewer.get("pdf_summary", {})
        if pdf_summary:
            lines.append(
                f"   PDFs: {pdf_summary.get('total_with_pdf', 0)}/{pdf_summary.get('total_publications', 0)}"
            )
            recovered = pdf_summary.get("recovered_by_source", {})
            failures = pdf_summary.get("failures_by_source", {})
            if recovered:
                recovered_text = ", ".join(
                    f"{src}:{count}" for src, count in sorted(recovered.items())
                )
                lines.append(f"   Recovered: {recovered_text}")
            if failures:
                failures_text = ", ".join(
                    f"{src}:{count}" for src, count in sorted(failures.items())
                )
                lines.append(f"   Recovery failures: {failures_text}")
        for paper in reviewer["papers"]:
            pub = paper["paper"]
            title = pub.get("title") or "Untitled"
            year = pub.get("year") or ""
            venue = pub.get("venue") or ""
            score = paper["score"]
            suffix = f" ({year})" if year else ""
            venue_text = f" {venue}" if venue else ""
            lines.append(f"   - {title}{suffix}{venue_text} [score {score:.4f}]")
    return "\n".join(lines)


def render_markdown(ranked_reviewers, top_n):
    lines = []
    for idx, reviewer in enumerate(ranked_reviewers[:top_n], 1):
        lines.append(f"{idx}. **{reviewer['name']}** - Score: {reviewer['score']:.4f}")
        pdf_summary = reviewer.get("pdf_summary", {})
        if pdf_summary:
            lines.append(
                f"   - PDFs: {pdf_summary.get('total_with_pdf', 0)}/"
                f"{pdf_summary.get('total_publications', 0)}"
            )
            recovered = pdf_summary.get("recovered_by_source", {})
            failures = pdf_summary.get("failures_by_source", {})
            if recovered:
                recovered_text = ", ".join(
                    f"{src}:{count}" for src, count in sorted(recovered.items())
                )
                lines.append(f"   - Recovered: {recovered_text}")
            if failures:
                failures_text = ", ".join(
                    f"{src}:{count}" for src, count in sorted(failures.items())
                )
                lines.append(f"   - Recovery failures: {failures_text}")
        for paper in reviewer["papers"]:
            pub = paper["paper"]
            title = pub.get("title") or "Untitled"
            year = pub.get("year") or ""
            venue = pub.get("venue") or ""
            score = paper["score"]
            suffix = f" ({year})" if year else ""
            venue_text = f" {venue}" if venue else ""
            lines.append(f"   - {title}{suffix}{venue_text} (score {score:.4f})")
    return "\n".join(lines)


def render_json(ranked_reviewers, top_n):
    return json.dumps({"reviewers": ranked_reviewers[:top_n]}, indent=2)

# ----------- Main Function ----------- #

def match_reviewers(
    cfp_url,
    submission_pdf,
    sources,
    text_source,
    output_format,
    top_n=8,
    top_papers=3,
    min_pdf=0,
):
    print("Extracting PC members...")
    pc_list = extract_pc_members(cfp_url)
    print(f"Found {len(pc_list)} PC members")

    print("Building reviewer profiles...")
    profiles, summary = build_reviewer_profiles(pc_list, sources, min_pdf)

    print("Parsing submission PDF...")
    paper_text = extract_text_from_pdf(submission_pdf)
    if not paper_text.strip():
        raise ValueError(
            "Failed to extract text from submission PDF. "
            "Install markitdown[pdf] or use a different PDF."
        )

    print("Computing similarity scores...")
    ranked_reviewers = compute_similarity(
        profiles, paper_text, text_source, top_papers=top_papers
    )

    if output_format == "json":
        output = render_json(ranked_reviewers, top_n)
    elif output_format == "markdown":
        output = render_markdown(ranked_reviewers, top_n)
    else:
        output = render_table(ranked_reviewers, top_n)

    print(output)
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
