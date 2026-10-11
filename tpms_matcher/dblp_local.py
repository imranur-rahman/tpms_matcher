# dblp_local.py
#
# Offline DBLP source. Builds a SQLite index from the DBLP XML dump
# (https://dblp.org/xml/dblp.xml.gz, with dblp.dtd next to it) and answers the
# author and publication queries that the matcher needs. The dblp.org API now
# answers scripts with a bot check page, so the dump is the reliable way to use DBLP.

import gzip
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

DEFAULT_DUMP_PATH = Path("dblp_data") / "dblp.xml.gz"
DEFAULT_INDEX_PATH = Path("dblp_data") / "dblp_index.sqlite"
PUBLICATION_TAGS = ("article", "inproceedings", "incollection")
PERSON_TAG = "www"
MIN_YEAR = 1990
BATCH_SIZE = 50000
HOMONYM_SUFFIX = re.compile(r"\s+\d{4}$")
ORCID_PATTERN = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]")
# Person records also link to profiles on other sites; these are not homepages
NOT_HOMEPAGE_HOSTS = (
    "orcid.org", "scholar.google", "wikidata.org", "dblp.org", "dblp.uni-trier.de",
    "researchgate.net", "linkedin.com", "twitter.com", "x.com", "genealogy.math",
    "wikipedia.org", "acm.org", "ieee.org", "scopus.com", "github.com", "zbmath.org",
    "semanticscholar.org", "isni.org", "viaf.org", "loop.frontiersin.org", "d-nb.info",
    "id.loc.gov", "mathscinet", "openreview.net", "bsky.app", "mastodon", "facebook.com",
)


def pick_homepage(urls):
    """Return the first link of a DBLP person record that is a personal or lab page"""
    for url in urls:
        url = (url or "").strip()
        if url.startswith("http") and not any(host in url for host in NOT_HOMEPAGE_HOSTS):
            return url
    return None

SCHEMA = """
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    norm TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pubs (
    id INTEGER PRIMARY KEY,
    dblp_key TEXT,
    title TEXT,
    year INTEGER,
    venue TEXT,
    doi TEXT,
    url TEXT
);
CREATE TABLE IF NOT EXISTS authorship (
    author_id INTEGER NOT NULL,
    pub_id INTEGER NOT NULL,
    orcid TEXT
);
CREATE TABLE IF NOT EXISTS persons (
    author_id INTEGER NOT NULL,
    primary_id INTEGER NOT NULL,
    affiliation TEXT,
    orcid TEXT,
    dblp_key TEXT,
    homepage TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_authors_norm ON authors(norm);
CREATE INDEX IF NOT EXISTS idx_authorship_author ON authorship(author_id);
CREATE INDEX IF NOT EXISTS idx_authorship_pub ON authorship(pub_id);
CREATE INDEX IF NOT EXISTS idx_authorship_orcid ON authorship(orcid);
CREATE INDEX IF NOT EXISTS idx_persons_author ON persons(author_id);
CREATE INDEX IF NOT EXISTS idx_persons_primary ON persons(primary_id);
CREATE INDEX IF NOT EXISTS idx_persons_orcid ON persons(orcid);
CREATE INDEX IF NOT EXISTS idx_persons_key ON persons(dblp_key);
"""


def fold_name(name):
    """Lowercase, remove accents, the DBLP homonym number, and punctuation"""
    name = HOMONYM_SUFFIX.sub("", name or "")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _text(elem):
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def build_index(dump_path=DEFAULT_DUMP_PATH, index_path=DEFAULT_INDEX_PATH):
    """Stream the DBLP dump into a SQLite index. The dump needs dblp.dtd in the same folder."""
    from lxml import etree

    dump_path = Path(dump_path)
    index_path = Path(index_path)
    if not dump_path.exists():
        raise FileNotFoundError(f"DBLP dump not found: {dump_path}")
    if not (dump_path.parent / "dblp.dtd").exists():
        raise FileNotFoundError(f"dblp.dtd must be in the same folder as {dump_path}")

    tmp_path = index_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    conn = sqlite3.connect(str(tmp_path))
    conn.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    conn.executescript(SCHEMA)

    author_ids = {}
    pub_rows, authorship_rows, person_rows = [], [], []
    next_pub_id = 1
    start = time.time()

    def author_id(name):
        aid = author_ids.get(name)
        if aid is None:
            aid = len(author_ids) + 1
            author_ids[name] = aid
        return aid

    def flush():
        conn.executemany("INSERT INTO pubs VALUES (?,?,?,?,?,?,?)", pub_rows)
        conn.executemany("INSERT INTO authorship VALUES (?,?,?)", authorship_rows)
        conn.executemany("INSERT INTO persons VALUES (?,?,?,?,?,?)", person_rows)
        conn.commit()
        pub_rows.clear()
        authorship_rows.clear()
        person_rows.clear()

    class DtdResolver(etree.Resolver):
        # The dump refers to "dblp.dtd"; it defines the entities such as &ouml;
        def resolve(self, system_url, public_id, context):
            if system_url and system_url.endswith(".dtd"):
                return self.resolve_filename(str(dump_path.parent / "dblp.dtd"), context)
            return None

    source = gzip.open(dump_path, "rb") if dump_path.suffix == ".gz" else open(dump_path, "rb")
    context = etree.iterparse(
        source,
        events=("end",),
        tag=PUBLICATION_TAGS + (PERSON_TAG,),
        load_dtd=True,
        resolve_entities=True,
        huge_tree=True,
    )
    context.resolvers.add(DtdResolver())
    for _, elem in context:
        key = elem.get("key", "")
        if elem.tag == PERSON_TAG:
            if key.startswith("homepages/"):
                names = [_text(a) for a in elem.findall("author")]
                if names:
                    primary = author_id(names[0])
                    affiliations = [
                        _text(n) for n in elem.findall("note") if n.get("type") == "affiliation"
                    ]
                    urls = [url.text or "" for url in elem.findall("url")]
                    orcid = ""
                    for url in urls:
                        match = ORCID_PATTERN.search(url)
                        if "orcid.org" in url and match:
                            orcid = match.group().upper()
                            break
                    homepage = pick_homepage(urls)
                    for name in names:
                        person_rows.append(
                            (
                                author_id(name),
                                primary,
                                "; ".join(affiliations) or None,
                                orcid or None,
                                key,
                                homepage,
                            )
                        )
        else:
            year_elem = elem.find("year")
            year = int(year_elem.text) if year_elem is not None and (year_elem.text or "").isdigit() else None
            authors = elem.findall("author")
            title_elem = elem.find("title")
            if authors and title_elem is not None and year and year >= MIN_YEAR:
                venue_elem = elem.find("booktitle")
                if venue_elem is None:
                    venue_elem = elem.find("journal")
                doi = None
                url = None
                for ee in elem.findall("ee"):
                    link = ee.text or ""
                    if "doi.org/" in link and not doi:
                        doi = link.split("doi.org/", 1)[1].lower()
                    if not url:
                        url = link
                pub_rows.append(
                    (
                        next_pub_id,
                        key,
                        _text(title_elem).rstrip("."),
                        year,
                        _text(venue_elem) if venue_elem is not None else None,
                        doi,
                        url,
                    )
                )
                for author in authors:
                    orcid = author.get("orcid")
                    authorship_rows.append((author_id(_text(author)), next_pub_id, orcid.upper() if orcid else None))
                next_pub_id += 1

        # Free the parsed element and its finished siblings to keep memory flat
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if len(pub_rows) >= BATCH_SIZE:
            flush()
            if next_pub_id % 500000 < BATCH_SIZE:
                print(f"  {next_pub_id - 1} publications indexed ({time.time() - start:.0f}s)")

    source.close()
    flush()
    conn.executemany(
        "INSERT INTO authors VALUES (?,?,?)",
        ((aid, name, fold_name(name)) for name, aid in author_ids.items()),
    )
    conn.commit()
    print("Creating indexes...")
    conn.executescript(INDEXES)
    conn.close()
    tmp_path.replace(index_path)
    print(
        f"DBLP index written to {index_path}: {next_pub_id - 1} publications, "
        f"{len(author_ids)} author names ({time.time() - start:.0f}s)"
    )
    return index_path


class DblpIndex:
    """Read-only queries on an index that build_index made"""

    def __init__(self, index_path=DEFAULT_INDEX_PATH):
        self.path = Path(index_path)
        self._local = threading.local()

    @property
    def conn(self):
        # SQLite connections must not be shared between threads; open one per thread
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            self._local.conn = conn
        return conn

    def _primary(self, author_id):
        row = self.conn.execute(
            "SELECT primary_id FROM persons WHERE author_id = ? LIMIT 1", (author_id,)
        ).fetchone()
        return row[0] if row else author_id

    def _aliases(self, primary_id):
        rows = self.conn.execute(
            "SELECT author_id FROM persons WHERE primary_id = ?", (primary_id,)
        ).fetchall()
        return sorted({primary_id} | {r[0] for r in rows})

    @property
    def has_homepages(self):
        """Indexes from before the homepage column exist; they still work without homepages"""
        if not hasattr(self, "_has_homepages"):
            columns = [row[1] for row in self.conn.execute("PRAGMA table_info(persons)")]
            self._has_homepages = "homepage" in columns
        return self._has_homepages

    def describe(self, primary_id):
        name = self.conn.execute("SELECT name FROM authors WHERE id = ?", (primary_id,)).fetchone()
        homepage_column = "homepage" if self.has_homepages else "NULL"
        person = self.conn.execute(
            f"SELECT affiliation, orcid, {homepage_column} FROM persons WHERE primary_id = ? "
            "AND (affiliation IS NOT NULL OR orcid IS NOT NULL OR "
            f"{homepage_column} IS NOT NULL) LIMIT 1",
            (primary_id,),
        ).fetchone()
        aliases = self._aliases(primary_id)
        placeholders = ",".join("?" * len(aliases))
        count = self.conn.execute(
            f"SELECT COUNT(*) FROM authorship WHERE author_id IN ({placeholders})", aliases
        ).fetchone()[0]
        return {
            "id": primary_id,
            "name": name[0] if name else "",
            "affiliation": person[0] if person else None,
            "orcid": person[1] if person else None,
            "homepage": person[2] if person else None,
            "publication_count": count,
        }

    def find_by_orcid(self, orcid):
        orcid = (orcid or "").upper()
        if not orcid:
            return None
        row = self.conn.execute(
            "SELECT primary_id FROM persons WHERE orcid = ? LIMIT 1", (orcid,)
        ).fetchone()
        if row:
            return self.describe(row[0])
        row = self.conn.execute(
            "SELECT author_id, COUNT(*) AS n FROM authorship WHERE orcid = ? "
            "GROUP BY author_id ORDER BY n DESC LIMIT 1",
            (orcid,),
        ).fetchone()
        return self.describe(self._primary(row[0])) if row else None

    def find_by_name(self, name):
        """Return every DBLP person whose name (any alias) matches, without the homonym number"""
        rows = self.conn.execute("SELECT id FROM authors WHERE norm = ?", (fold_name(name),)).fetchall()
        primaries = {self._primary(r[0]) for r in rows}
        return [self.describe(p) for p in sorted(primaries)]

    def find_by_dblp_url(self, url):
        """Resolve a researchr 'dblp' profile link (https://dblp.org/pid/... or /pers/hd/...)"""
        match = re.search(r"/pid/([^?#]+?)(?:\.html|\.xml)?(?:[?#].*)?$", url or "")
        if not match:
            return None
        row = self.conn.execute(
            "SELECT primary_id FROM persons WHERE dblp_key = ? LIMIT 1",
            (f"homepages/{match.group(1)}",),
        ).fetchone()
        return self.describe(row[0]) if row else None

    def publications(self, primary_id):
        aliases = self._aliases(primary_id)
        placeholders = ",".join("?" * len(aliases))
        rows = self.conn.execute(
            f"SELECT DISTINCT p.id, p.dblp_key, p.title, p.year, p.venue, p.doi, p.url "
            f"FROM authorship a JOIN pubs p ON p.id = a.pub_id WHERE a.author_id IN ({placeholders})",
            aliases,
        ).fetchall()
        publications = []
        for pub_id, key, title, year, venue, doi, url in rows:
            authors = [
                HOMONYM_SUFFIX.sub("", r[0])
                for r in self.conn.execute(
                    "SELECT au.name FROM authorship a JOIN authors au ON au.id = a.author_id "
                    "WHERE a.pub_id = ?",
                    (pub_id,),
                )
            ]
            publications.append(
                {
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "url": f"https://dblp.org/rec/{key}" if key else url,
                    "authors": authors,
                    "abstract": "",
                    "pdf_url": "",
                    "doi": doi,
                    "arxiv_id": "",
                    "sources": ["dblp"],
                }
            )
        return publications
