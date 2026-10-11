# TPMS Matcher
Find similar papers to a submission by comparing it to PC members' published papers. This repo is built on top of to4grep repo.

## Installation
```bash
git clone https://github.com/imranur-rahman/tpms_matcher
cd tpms_matcher
pip3 install -e .
```

### API keys
Reviewer matching (`--match-reviewers`) reads publications from OpenAlex and Semantic Scholar. The paper search (`-k`) does not need these keys.

| Variable | Service | Needed? | Without it | Get it |
|---|---|---|---|---|
| `OPENALEX_API_KEY` | OpenAlex | **Yes**, for any PC of real size | OpenAlex gives a budget of $0.10 per day (about 1,000 requests) that all users of your IP address share. The tool sends 2–6 requests per PC member, so the budget runs out during one run. Then the tool skips OpenAlex for the rest of the run. | Free: https://openalex.org/settings/api |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar | Recommended | Requests go to a pool that all users without a key share. Some requests fail with HTTP 429, and the members of those requests get no Semantic Scholar papers. | Free, after a request form: https://www.semanticscholar.org/product/api |
| `OPENALEX_EMAIL` | OpenAlex | Optional | Nothing changes. The tool adds the address to OpenAlex requests (`mailto`). | Your e-mail address |

The tool sends at most 1 request per second to Semantic Scholar, with or without a key, because that is the limit for a key.

Set the keys in `~/.zshenv` (zsh) or `~/.bashrc` (bash). Do not use `~/.zshrc` for zsh: scripts and non-interactive shells do not read it.
```bash
echo 'export OPENALEX_API_KEY=<your key>' >> ~/.zshenv
echo 'export SEMANTIC_SCHOLAR_API_KEY=<your key>' >> ~/.zshenv
source ~/.zshenv
```

Check the OpenAlex key (the answer must be `HTTP 200`):
```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" -H "Authorization: Bearer $OPENALEX_API_KEY" \
  "https://api.openalex.org/authors?search=test&per_page=1"
```

At the end of a run, the tool prints how many members lost a source (`Members with a failed source`). If that line shows `openalex` or `semanticscholar`, check the key.

## Usage
### Database Initialization (Similar to top4grep)
Build the paper database with optional conference type selection or update the database of papers stored in `papers.db`:

```bash
# Build database for all conferences (default)
tpms-matcher --build-db

# Build database for security conferences only
tpms-matcher --build-db --conference-type security

# Build database for software engineering conferences only
tpms-matcher --build-db --conference-type software_engineering
```

Supported conferences:
- Security: NDSS, IEEE S&P, USENIX, CCS
- Software Engineering: ICSE, FSE, ASE, ISSTA

### Query (Similar to top4grep)
```bash
tpms-matcher -k <keywords> [--start-year YEAR]
```

Examples:
```bash
# Search for papers with keywords from all years (default: 2000)
tpms-matcher -k linux,kernel

# Search for papers from 2015 onwards
tpms-matcher -k linux,kernel --start-year 2015
```
To return all paper titles, don't include the k argument.
```bash
tpms-matcher --start-year 2010 --conference-type software_engineering
```

The query performs a case-insensitive match (like grep). The returned results must contain all input keywords (papers containing keyword1 AND keyword2 AND ...). Support for `OR` operation (papers containing keyword1 OR keyword2) is planned for future updates.

### TPMS-Style Reviewer Matching
```bash
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH>
```

`CFP_URL` can be a researchr track page (`/track/...`), a researchr committee page (`/committee/...`), a HotCRP site, or a local HTML file that you saved from the browser.

Options:
```bash
# Choose publication sources (comma-separated)
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --sources dblp,openalex,semanticscholar,website

# Choose text source for similarity
#   auto: title + abstract; reads the first 2 PDF pages only when no abstract is known
#   pdf: title + up to 3000 words of the PDF; abstract: title + abstract; title: title only
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --text-source auto

# Choose the similarity model: tfidf (default), lsa, embedding (SPECTER)
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --scorer tfidf

# Change the score: score = alpha * top-k part + (1 - alpha) * whole-profile part
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --top-k 3 --alpha 0.5 --half-life 5

# Check conflicts of interest (flag them, or remove the reviewers with --coi exclude)
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --authors "Ada Lovelace;Alan Turing" \
    --author-affiliations "University of Manchester" --coi-years 2 --coi flag

# Output formats: table (default), json, markdown
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --output json

# Control display sizes
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --top-n 8 --top-papers 3

# Filter reviewers by minimum PDF availability
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --min-pdf 5

# Read the PC list from HotCRP when the researchr page has no PC members yet
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --hotcrp-url https://<conf>.hotcrp.com

# Use the HotCRP site directly as the PC source
tpms-matcher --match-reviewers https://<conf>.hotcrp.com <PDF_PATH>

# researchr profiles: lazy (default), all, or none
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --profiles lazy --profile-top 20

# Quick test with the first 20 PC members only
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --max-members 20
```

Install the embedding scorer (SPECTER, through sentence-transformers) with `pip3 install -e '.[embeddings]'`.
TF-IDF is the default because it passed a self-check that SPECTER failed: on the FSE 2027 PC, with an arXiv paper as the submission, TF-IDF ranked the paper's own authors 1–3, and SPECTER ranked them 10th, 18th, and 98th.

#### How the matcher works
1. **PC members.** The tool reads the researchr page. If the page has no members, it uses the HotCRP site from `--hotcrp-url`. HotCRP shows the PC list at `<site>/users/pc` only when the chairs make it public.
2. **Profiles.** The researchr list page gives name, affiliation, country, and role. The DBLP index gives the homepage and the ORCID iD when DBLP identifies the member by ORCID iD, profile link, affiliation, or a unique name. With `--profiles lazy`, the tool fetches researchr profiles only for the members that no source identified by ORCID iD or affiliation, and for the top `--profile-top` reviewers. Then it collects their publications again and ranks again.
3. **Author identity.** For each source, the tool picks the author record by ORCID iD first, then by affiliation, then by a unique name. The output shows the method per source. `name-only` means that more than one person has the name and the tool could not tell them apart. When another source identified the member better, the tool keeps a `name-only` paper only if that source has the same title, so a namesake's papers stay out.
4. **Publications.** DBLP, OpenAlex (all pages, up to 1000 works), and Semantic Scholar (up to 1000 papers, with abstracts). The tool merges the records of one paper by DOI or title, so a preprint and its published version count once. Website entries only add PDF links and abstracts to papers that an API found.
5. **Text.** Each paper is title + abstract, limited to 400 words, so all papers have a similar length. The tool removes the reference list and template text (ACM reference format, CCS concepts, copyright lines) from the submission.
6. **Score.** `alpha * (sum of the top-k decayed similarities / k) + (1 - alpha) * (similarity to the decayed mean of all papers)`. A paper that is `half-life` years old counts half. A reviewer with fewer than k papers gets 0 for the missing papers.

#### DBLP
The dblp.org API now answers scripts with a bot check page. Use a local index of the DBLP dump instead:
```bash
# Download https://dblp.org/xml/dblp.xml.gz and https://dblp.org/xml/dblp.dtd into dblp_data/, then:
tpms-matcher --build-dblp-index            # writes dblp_data/dblp_index.sqlite (about 5 minutes, 3.6 GB)
# Rebuild an index from before October 2026 to get the homepage column
```
The matcher uses `dblp_data/dblp_index.sqlite` when it exists, or the file from `--dblp-index`.

#### Rate limits and caches
- researchr bans clients that fetch pages too fast (its `robots.txt` sets `Crawl-delay: 2`). The tool fetches profiles one at a time, waits the crawl delay of the site (at least 2 s), obeys the `Disallow` rules, and caches pages for 7 days in `~/.cache/tpms_matcher` (change it with `TPMS_CACHE_DIR`). If researchr shows a captcha, the tool stops fetching profiles. Save the list page in your browser and pass the file path.
- The tool caches PDF text and SPECTER vectors in the same folder.
- Set `OPENALEX_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` (see [API keys](#api-keys)).
- API answers are cached for 7 days, so a repeated run sends few requests. When an API asks to wait more than 2 minutes (a used-up daily budget), the tool skips that source for the rest of the run and says so in the output.

#### Tests
```bash
pip3 install -e '.[test]'
pytest tests
```

## TODO
- [ ] grep in abstract
- [ ] fuzzy match
- [ ] complex search logic (`OR` operation)
