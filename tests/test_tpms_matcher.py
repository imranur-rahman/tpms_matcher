import gzip

import pytest

from tpms_matcher import tpms_matcher as tm
from tpms_matcher import dblp_local


def pub(title, year=2024, **extra):
    record = {
        "title": title, "year": year, "venue": None, "url": "", "authors": [],
        "abstract": "", "pdf_url": "", "doi": "", "arxiv_id": "", "sources": ["test"],
    }
    record.update(extra)
    return record


# ----------- PC member extraction ----------- #

HOTCRP_HTML = """
<table class="pltable"><tbody>
<tr><td class="pl pl_name"><span class="taghl">Ada Lovelace</span> <span class="pcrole">chair</span></td>
<td class="pl pl_affiliation">University of London</td><td class="pl pl_orcid">0000-0002-1825-009X</td>
<td class="pl pl_country">United Kingdom</td></tr>
<tr><td class="pl pl_name"><span class="taghl">Alan Turing</span></td>
<td class="pl pl_affiliation">University of Manchester</td><td class="pl pl_orcid"></td><td class="pl pl_country"></td></tr>
</tbody></table>
"""

RESEARCHR_TRACK_HTML = """
<ul class="list-group"><li class="list-group-item"><a class="navigate" href="/profile/fse-2027/alberto">
<div class="media"><div class="media-body">
<h5 class="media-heading">Alberto Bacchelli <span class="pull-right"><small>Area Chair</small></span></h5>
<h5 class="media-heading"><span class="text-black">IfI, University of Zurich</span></h5>
<h5 class="media-heading"><small>Switzerland</small></h5>
</div></div></a></li></ul>
"""

RESEARCHR_COMMITTEE_HTML = """
<div class="row"><div class="col-sm-6"><a class="navigate" href="https://conf.researchr.org/profile/fse-2027/michaelpradel">
<div class="media"><div class="media-body">
<h3 class="media-heading">Michael Pradel <small>Chair</small></h3>
<h4 class="media-heading"><span class="text-black">CISPA Helmholtz Center for Information Security</span></h4>
<h4 class="media-heading"><small>Germany</small></h4>
</div></div></a></div></div>
<a class="navigate" href="https://conf.researchr.org/profile/fse-2027/michaelpradel">Michael Pradel</a>
"""


def test_parse_hotcrp_pc_page():
    members = tm.parse_hotcrp_pc_page(HOTCRP_HTML)
    assert [m["name"] for m in members] == ["Ada Lovelace", "Alan Turing"]
    assert members[0]["role"] == "chair"
    assert members[0]["orcid"] == "0000-0002-1825-009X"
    assert members[0]["country"] == "United Kingdom"
    assert "orcid" not in members[1]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://conf.hotcrp.com", "https://conf.hotcrp.com/users/pc"),
        ("https://conf.hotcrp.com/users?t=pc", "https://conf.hotcrp.com/users/pc"),
        ("https://conf.hotcrp.com/u/0/paper/3", "https://conf.hotcrp.com/users/pc"),
        ("https://host.org/conf26/u/1/users/pc", "https://host.org/conf26/users/pc"),
    ],
)
def test_hotcrp_pc_url(url, expected):
    assert tm.hotcrp_pc_url(url) == expected


def test_parse_researchr_track_layout_keeps_comma_in_affiliation():
    members = tm.parse_researchr_members(RESEARCHR_TRACK_HTML, "https://conf.researchr.org/track/x")
    assert len(members) == 1
    member = members[0]
    assert member["name"] == "Alberto Bacchelli"
    assert member["role"] == "Area Chair"
    assert member["affiliation"] == "IfI, University of Zurich"
    assert member["country"] == "Switzerland"
    assert member["url"] == "https://conf.researchr.org/profile/fse-2027/alberto"


def test_parse_researchr_committee_layout():
    members = tm.parse_researchr_members(RESEARCHR_COMMITTEE_HTML, "https://conf.researchr.org/")
    assert len(members) == 1
    assert members[0]["name"] == "Michael Pradel"
    assert members[0]["role"] == "Chair"
    assert members[0]["affiliation"] == "CISPA Helmholtz Center for Information Security"
    assert members[0]["country"] == "Germany"


def test_generic_fallback_keeps_only_person_names():
    html = """
    <ul><li>Program Committee</li><li>Important Dates</li>
    <li>Jane Doe, MIT</li><li>Ludwig van Beethoven (Bonn University)</li>
    <li>Submission deadline: 2026-10-02</li><li>Jane Doe, MIT</li></ul>
    """
    members = tm.parse_generic_members(html, "https://example.org/")
    assert [m["name"] for m in members] == ["Jane Doe", "Ludwig van Beethoven"]
    assert members[0]["affiliation"] == "MIT"


# ----------- Identity ----------- #

def test_names_match():
    assert tm.names_match("Marcel Böhme", "Marcel Bohme")
    assert tm.names_match("J. Smith", "John Smith")
    assert tm.names_match("Lei Zhang 0001", "lei zhang")
    assert not tm.names_match("Lei Zhang", "Lei Wang")
    assert not tm.names_match("Jane Smith", "John Smith")


def test_pick_author_prefers_orcid_then_affiliation():
    candidates = [
        {"name": "Lei Zhang", "affiliations": ["Fudan University"], "orcid": None, "count": 900},
        {"name": "Lei Zhang", "affiliations": ["University of Illinois Urbana-Champaign"], "orcid": "0000-0001-0000-0001", "count": 50},
    ]
    chosen, method = tm.pick_author(candidates, {"name": "Lei Zhang", "orcid": "https://orcid.org/0000-0001-0000-0001"})
    assert method == "orcid" and chosen is candidates[1]

    member = {"name": "Lei Zhang", "affiliation": "University of Illinois at Urbana-Champaign"}
    chosen, method = tm.pick_author(candidates, member)
    assert method == "affiliation" and chosen is candidates[1]

    chosen, method = tm.pick_author(candidates, {"name": "Lei Zhang"})
    assert method == "name-only" and chosen is candidates[0]

    chosen, method = tm.pick_author(candidates, {"name": "Someone Else"})
    assert chosen is None


# ----------- Publications ----------- #

def test_dedupe_joins_preprint_and_published_version():
    pubs = [
        pub("Fuzzing the Web", 2023, doi="10.48550/arXiv.2301.0001", sources=["openalex"]),
        pub("Fuzzing the web.", 2024, doi="10.1145/123", sources=["dblp"]),
        pub("Another Paper", 2024, doi="10.1145/123", abstract="long abstract", sources=["semanticscholar"]),
        pub("Unrelated Paper", 2020),
    ]
    merged = tm.dedupe_publications(pubs)
    assert len(merged) == 2
    assert merged[0]["year"] == 2024
    assert merged[0]["sources"] == ["dblp", "openalex", "semanticscholar"]
    assert merged[0]["abstract"] == "long abstract"


def test_parse_bibtex_with_nested_braces():
    bib = r"""
    @inproceedings{key1,
      title = {{BERT}-based {T}est {G}eneration for {C}++},
      author = {M{\"o}ller, Anna and Smith, Bob},
      year = 2024,
      booktitle = "Proc. of {ICSE}",
      doi = {10.1145/999}
    }
    @comment{ignored}
    @article(key2, title="Plain Title", year={2019})
    """
    pubs = tm.parse_bibtex_publications(bib)
    assert [p["title"] for p in pubs] == ["BERT-based Test Generation for C++", "Plain Title"]
    assert pubs[0]["year"] == 2024
    assert pubs[0]["authors"] == ["Moller, Anna", "Smith, Bob"]
    assert pubs[0]["venue"] == "Proc. of ICSE"
    assert pubs[1]["year"] == 2019


def test_attach_website_publications_matches_titles_only():
    api_pubs = [pub("Fuzzing the Web", 2024)]
    website = [
        pub("Fuzzing the web", 2009, pdf_url="https://me.org/fuzz.pdf", sources=["website"]),
        pub("News: our lab won an award this year", 2025, sources=["website"]),
    ]
    result = tm.attach_website_publications(api_pubs, website, [])
    assert len(result) == 1
    assert result[0]["pdf_url"] == "https://me.org/fuzz.pdf"
    assert result[0]["year"] == 2024
    assert "website" in result[0]["sources"]
    # Without API results the website entries are the only evidence
    assert len(tm.attach_website_publications([], website, [])) == 2


# ----------- Text ----------- #

SUBMISSION = """Neural Fuzzing for Compilers
Anonymous Author(s)
ABSTRACT
We fuzz compilers with a neural model that writes test programs.
CCS Concepts: Software and its engineering
1 INTRODUCTION
Compilers have bugs. """ + "Compiler testing finds crashes. " * 50 + """
REFERENCES
[1] A. Smith. Linux kernel exploits. 2010.
[2] B. Jones. Kernel exploitation. 2012.
"""


def test_clean_paper_text_removes_references_and_template():
    cleaned = tm.clean_paper_text(SUBMISSION)
    assert "kernel" not in cleaned.lower()
    assert "CCS Concepts" not in cleaned
    assert "Anonymous" not in cleaned
    assert "Compiler testing" in cleaned


def test_extract_abstract():
    assert tm.extract_abstract(SUBMISSION) == "We fuzz compilers with a neural model that writes test programs."


def test_publication_document_has_bounded_length():
    long_abstract = "word " * 5000
    doc = tm.publication_document(pub("Title", abstract=long_abstract), "abstract")
    assert len(doc.split()) == tm.DOC_WORD_LIMIT


def test_decay_weight():
    assert tm.exponential_decay_weight(2020, current_year=2025, half_life=5) == pytest.approx(0.5)
    assert tm.exponential_decay_weight(2027, current_year=2025, half_life=5) == 1.0
    assert tm.exponential_decay_weight(None) == tm.UNKNOWN_YEAR_WEIGHT
    assert tm.exponential_decay_weight(1990, half_life=0) == 1.0


# ----------- Scoring ----------- #

def make_profiles():
    return {
        "Fuzz Expert": {"publications": [
            pub("Compiler fuzzing with neural program generation", 2025),
            pub("Fuzzing compilers to find crashes", 2024),
            pub("Neural test program generation for compiler testing", 2023),
        ]},
        "One Hit": {"publications": [pub("Compiler fuzzing with neural program generation", 2025)]},
        "Old Expert": {"publications": [
            pub("Compiler fuzzing with neural program generation", 1995),
            pub("Fuzzing compilers to find crashes", 1994),
            pub("Neural test program generation for compiler testing", 1993),
        ]},
        "Database Person": {"publications": [
            pub("Query optimization for column stores", 2025),
            pub("Transaction processing at scale", 2024),
        ]},
    }


def test_compute_similarity_ranks_breadth_and_recency():
    ranked = tm.compute_similarity(
        make_profiles(), SUBMISSION, "title", scorer="tfidf", top_k=3, alpha=0.5, half_life=5
    )
    names = [r["name"] for r in ranked]
    # A reviewer with one close paper no longer ties with a reviewer with three
    assert names.index("Fuzz Expert") < names.index("One Hit")
    # The same papers 30 years ago count less
    assert names.index("Fuzz Expert") < names.index("Old Expert")
    assert names[-1] == "Database Person"
    for reviewer in ranked:
        assert reviewer["score"] == pytest.approx(0.5 * reviewer["top_k_score"] + 0.5 * reviewer["profile_score"])


@pytest.mark.parametrize("scorer", ["tfidf", "lsa"])
def test_compute_similarity_scorers_run(scorer):
    ranked = tm.compute_similarity(make_profiles(), SUBMISSION, "title", scorer=scorer)
    assert ranked[0]["name"] != "Database Person"


# ----------- Conflicts ----------- #

def test_detect_conflicts():
    member = {"name": "Ada Lovelace", "affiliation": "University of London"}
    pubs = [
        pub("Recent joint work", tm.CURRENT_YEAR - 1, authors=["Ada Lovelace", "Charles Babbage"]),
        pub("Old joint work", tm.CURRENT_YEAR - 10, authors=["Ada Lovelace", "Alan Turing"]),
    ]
    conflicts = tm.detect_conflicts(member, pubs, ["C. Babbage", "Alan Turing"], ["Univ. of London"], 2)
    assert any("Charles Babbage" not in c and "C. Babbage" in c and "Recent joint work" in c for c in conflicts)
    assert not any("Old joint work" in c for c in conflicts)
    assert any("same affiliation" in c for c in conflicts)
    assert tm.detect_conflicts(member, pubs, ["Grace Hopper"], [], 2) == []


# ----------- DBLP ----------- #

DBLP_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE dblp SYSTEM "dblp.dtd">
<dblp>
<www key="homepages/12/345"><author>Marcel B&ouml;hme</author><title>Home Page</title>
<note type="affiliation">MPI-SP</note><url>https://orcid.org/0000-0002-4470-1824</url>
<url>https://scholar.google.com/citations?user=X</url><url>https://mboehme.github.io/</url></www>
<www key="homepages/99/1"><author>Lei Zhang 0001</author><title>Home Page</title>
<note type="affiliation">Fudan University</note></www>
<inproceedings key="conf/icse/B20"><author orcid="0000-0002-4470-1824">Marcel B&ouml;hme</author>
<author>Lei Zhang 0001</author><title>Fuzzing <i>Things</i>.</title><year>2020</year>
<booktitle>ICSE</booktitle><ee>https://doi.org/10.1145/ABC</ee></inproceedings>
<article key="journals/x/Z19"><author>Lei Zhang 0002</author><title>Other.</title><year>2019</year>
<journal>TSE</journal></article>
</dblp>"""


@pytest.fixture
def dblp_index(tmp_path):
    (tmp_path / "dblp.dtd").write_text(
        '<!ELEMENT dblp ANY>\n<!ENTITY ouml "&#246;">\n', encoding="latin-1"
    )
    with gzip.open(tmp_path / "dblp.xml.gz", "wb") as f:
        f.write(DBLP_XML.encode("latin-1"))
    dblp_local.build_index(tmp_path / "dblp.xml.gz", tmp_path / "index.sqlite")
    return dblp_local.DblpIndex(tmp_path / "index.sqlite")


def test_dblp_index_queries(dblp_index):
    by_orcid = dblp_index.find_by_orcid("0000-0002-4470-1824")
    assert by_orcid["name"] == "Marcel Böhme"
    assert dblp_index.find_by_dblp_url("https://dblp.org/pid/12/345.html")["name"] == "Marcel Böhme"
    assert {c["name"] for c in dblp_index.find_by_name("Lei Zhang")} == {"Lei Zhang 0001", "Lei Zhang 0002"}
    pubs = dblp_index.publications(by_orcid["id"])
    assert pubs[0]["title"] == "Fuzzing Things"
    assert pubs[0]["authors"] == ["Marcel Böhme", "Lei Zhang"]
    assert pubs[0]["doi"] == "10.1145/abc"


def test_fetch_publications_from_dblp_uses_affiliation(dblp_index):
    pubs, method = tm.fetch_publications_from_dblp(
        {"name": "Lei Zhang", "affiliation": "Fudan University"}, dblp_index
    )
    assert method == "affiliation"
    assert [p["title"] for p in pubs] == ["Fuzzing Things"]


def test_parse_dblp_person_xml():
    xml = """<?xml version="1.0"?><dblpperson name="X"><r><inproceedings key="conf/a/B1">
    <author pid="1">Ann Bee 0001</author><title>A <i>Title</i>.</title><year>2022</year>
    <booktitle>FSE</booktitle><ee>https://doi.org/10.1/XY</ee></inproceedings></r></dblpperson>"""
    pubs = tm.parse_dblp_person_xml(xml)
    assert pubs[0]["title"] == "A Title"
    assert pubs[0]["authors"] == ["Ann Bee"]
    assert pubs[0]["doi"] == "10.1/xy"


# ----------- Pages ----------- #

def test_block_page_detection(monkeypatch, tmp_path):
    captcha = "<html><head><title>CrowdSec Captcha</title></head><body>CrowdSec Captcha</body></html>"
    assert tm.is_block_page(captcha)
    assert not tm.is_block_page(RESEARCHR_TRACK_HTML)

    class Response:
        text = captcha
    monkeypatch.setattr(tm, "http_get", lambda url, **kw: Response())
    monkeypatch.setattr(tm, "HTML_CACHE_DIR", tmp_path)
    with pytest.raises(tm.BlockedPage):
        tm.fetch_html("https://conf.researchr.org/track/x", cache=True)
    assert not list(tmp_path.iterdir())  # the tool does not cache a captcha page


def test_extract_pc_members_reads_saved_page(tmp_path):
    page = tmp_path / "pc.html"
    page.write_text(RESEARCHR_COMMITTEE_HTML, encoding="utf-8")
    members = tm.extract_pc_members(str(page), fetch_profiles=False)
    assert [m["name"] for m in members] == ["Michael Pradel"]


@pytest.mark.parametrize(
    "title, expected",
    [
        ('Reproduction Package for "LLM-Based Repair of Static Nullability Errors"', False),
        ("CodeCureAgent (Code and Data)", False),
        ("PoCGen Evaluation Data", False),
        ("ponder-lab/ML: 0.42.0", False),
        ("ponder-lab/ML: Track initial tensor shapes and dtypes", False),
        ("Replication package for: a study of flaky tests", False),
        ("Names Are All You Need: Effective and Safe Regression Test Selection for Python", True),
        ("Artifact Evaluation in Software Engineering Conferences", True),
    ],
)
def test_is_research_paper(title, expected):
    assert tm.is_research_paper(pub(title)) is expected


def test_pick_homepage_skips_profile_sites():
    urls = ["https://orcid.org/0000-0002-4470-1824", "https://scholar.google.com/x", "https://mboehme.github.io/"]
    assert dblp_local.pick_homepage(urls) == "https://mboehme.github.io/"
    assert dblp_local.pick_homepage(["https://github.com/someone"]) is None


def test_dblp_match_fills_homepage_and_orcid(dblp_index):
    member = {"name": "Marcel Bohme", "affiliation": "MPI-SP"}
    pubs, method = tm.fetch_publications_from_dblp(member, dblp_index)
    assert method == "affiliation"
    assert member["personal_website"] == "https://mboehme.github.io/"
    assert member["orcid"] == "0000-0002-4470-1824"

    # A name-only match can be the wrong person, so it changes nothing
    namesake = {"name": "Lei Zhang"}
    tm.fetch_publications_from_dblp(namesake, dblp_index)
    assert "personal_website" not in namesake and "orcid" not in namesake


def test_site_rules_reads_crawl_delay(monkeypatch):
    monkeypatch.setattr(tm, "ROBOTS_CACHE", {})
    monkeypatch.setattr(
        tm, "fetch_html",
        lambda url, cache=False: "User-agent: *\nDisallow: /signin\nCrawl-delay: 5\n",
    )
    rules, delay = tm.site_rules("https://conf.researchr.org/profile/x/y")
    assert delay == 5
    assert rules.can_fetch("tpms_matcher", "https://conf.researchr.org/profile/x/y")
    assert not rules.can_fetch("tpms_matcher", "https://conf.researchr.org/signin")


def test_add_profile_details_waits_and_skips_fetched(monkeypatch):
    monkeypatch.setattr(tm, "site_rules", lambda url: (None, 3.0))
    monkeypatch.setattr(tm, "html_cache_path", lambda url: tm.Path("/nonexistent/cache"))
    monkeypatch.setattr(tm, "extract_profile_details", lambda url: {"personal_website": "https://a.org"})
    waits = []
    monkeypatch.setattr(tm.time, "sleep", waits.append)
    members = [{"name": "A B", "url": "https://r.org/profile/a"}, {"name": "C D", "url": "https://r.org/profile/c", "profile_fetched": True}]
    tm.add_profile_details(members)
    assert members[0]["personal_website"] == "https://a.org" and members[0]["profile_fetched"]
    assert "personal_website" not in members[1]
    assert waits == [3.0]


def test_cached_embeddings_encode_each_document_once(monkeypatch, tmp_path):
    monkeypatch.setattr(tm, "EMBEDDING_CACHE_PATH", tmp_path / "emb.sqlite")
    encoded = []

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, docs, **kwargs):
            encoded.extend(docs)
            return tm.np.array([[len(d), 1.0] for d in docs])

    first = tm.cached_embeddings(["aa", "bbb"], "m", FakeModel)
    second = tm.cached_embeddings(["bbb", "cccc"], "m", FakeModel)
    assert encoded == ["aa", "bbb", "cccc"]
    assert first[1].tolist() == second[0].tolist() == [3.0, 1.0]


def test_wait_for_host_spaces_requests(monkeypatch):
    monkeypatch.setattr(tm, "HOST_NEXT_REQUEST", {})
    clock = [100.0]
    waits = []
    monkeypatch.setattr(tm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tm.time, "sleep", waits.append)
    for _ in range(3):
        tm.wait_for_host("https://api.semanticscholar.org/graph/v1/author/search")
    tm.wait_for_host("https://example.org/")
    assert waits == [1.0, 2.0]


def test_fetch_json_uses_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tm, "API_CACHE_DIR", tmp_path)
    calls = []

    class Response:
        headers = {}

        def json(self):
            return {"n": len(calls)}

    def fake_get(url, params=None, headers=None):
        calls.append(url)
        return Response()

    monkeypatch.setattr(tm, "http_get", fake_get)
    assert tm.fetch_json("https://api.openalex.org/works", params={"a": 1}) == {"n": 1}
    assert tm.fetch_json("https://api.openalex.org/works", params={"a": 1}) == {"n": 1}
    assert tm.fetch_json("https://api.openalex.org/works", params={"a": 2}) == {"n": 2}
    assert len(calls) == 2


def test_long_retry_after_stops_the_source(monkeypatch):
    class Response:
        status_code = 429
        headers = {"Retry-After": "44798"}

    monkeypatch.setattr(tm.requests, "get", lambda *a, **kw: Response())
    monkeypatch.setattr(tm, "EXHAUSTED_SOURCES", {})
    monkeypatch.setattr(tm, "SOURCE_FAILURES", {})
    monkeypatch.setattr(tm, "API_CACHE_DIR", tm.Path("/nonexistent/cache"))
    def no_long_sleep(seconds):
        assert seconds <= tm.MAX_RETRY_WAIT, "the tool must not wait 12 hours"
    monkeypatch.setattr(tm.time, "sleep", no_long_sleep)
    with pytest.raises(tm.RateLimitExhausted):
        tm.http_get("https://api.openalex.org/authors")

    pubs, method = tm.fetch_publications_from_openalex({"name": "Ada Lovelace"})
    assert pubs == [] and method is None
    assert "openalex" in tm.EXHAUSTED_SOURCES
    # Later members skip the source without a request
    monkeypatch.setattr(tm.requests, "get", lambda *a, **kw: pytest.fail("no request expected"))
    assert tm.fetch_publications_from_openalex({"name": "Alan Turing"}) == ([], None)


def test_keep_only_confirmed_drops_namesake_papers():
    by_source = {
        "openalex": [pub("Fuzzing Deep Learning Libraries"), pub("Testing Tensor Compilers")],
        "semanticscholar": [
            pub("Fuzzing deep learning libraries", abstract="An abstract"),
            pub("Storage Impact on Chlamydospore Germination"),
        ],
    }
    kept = tm.keep_only_confirmed(by_source, {"openalex": "orcid", "semanticscholar": "name-only"})
    titles = [p["title"] for p in kept]
    assert "Storage Impact on Chlamydospore Germination" not in titles
    assert "Fuzzing deep learning libraries" in titles  # it still adds its abstract

    # Without a better-identified source, nothing can confirm or reject the papers
    kept = tm.keep_only_confirmed(by_source, {"openalex": "name-only", "semanticscholar": "name-only"})
    assert len(kept) == 4
