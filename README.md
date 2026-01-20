# TPMS Matcher
Find similar papers to a submission by comparing it to PC members' published papers.

## Installation
```bash
git clone https://github.com/imranur-rahman/tpms_matcher
cd tpms_matcher
pip3 install -e .
```

## Usage
### Database Initialization
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

### Query
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

Options:
```bash
# Choose publication sources (comma-separated)
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --sources dblp,openalex,semanticscholar,website

# Choose text source for similarity (auto uses pdf -> abstract -> title)
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --text-source auto

# Output formats: table (default), json, markdown
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --output json

# Control display sizes
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --top-n 8 --top-papers 3

# Filter reviewers by minimum PDF availability
tpms-matcher --match-reviewers <CFP_URL> <PDF_PATH> --min-pdf 5
```

## Screenshot
![screenshot](https://raw.githubusercontent.com/imranur-rahman/tpms_matcher/master/img/screenshot.png)

## TODO
- [ ] grep in abstract
- [ ] fuzzy match
- [ ] complex search logic (`OR` operation)
