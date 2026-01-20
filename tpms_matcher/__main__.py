import sqlalchemy
from sqlalchemy.orm import sessionmaker
from nltk import download, word_tokenize
from nltk.data import find
from nltk.stem import PorterStemmer

from .db import Base, Paper
from .build_db import build_db, DB_PATH, CONFERENCE_CATEGORIES
from .utils import new_logger
from .tpms_matcher import match_reviewers
import argparse



engine = sqlalchemy.create_engine(f'sqlite:///{str(DB_PATH)}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

logger = new_logger("Top4Grep")
stemmer = PorterStemmer()

CONFERENCES = CONFERENCE_CATEGORIES["all"]

# Function to check and download 'punkt' if not already available
def check_and_download_punkt():
    try:
        # Check if 'punkt_tab' is available, this will raise a LookupError if not found
        find('tokenizers/punkt_tab')
        #print("'punkt' tokenizer models are already installed.")
    except LookupError:
        print("'punkt_tab' tokenizer models not found. Downloading...")
        # Download 'punkt' tokenizer models
        download('punkt_tab')
        
# trim word tokens from tokenizer to stem i.e. exploiting to exploit
def fuzzy_match(title):
    tokens = word_tokenize(title)
    return [stemmer.stem(token) for token in tokens]

def existed_in_tokens(tokens, keywords):
    return all(map(lambda k: stemmer.stem(k.lower()) in tokens, keywords))

def grep(keywords, abstract, start_year=2000, conference_type='all'):
    # TODO: currently we only grep either from title or from abstract, also grep from other fields in the future maybe?
    # TODO: convert the pdf to markdown using markitdown and store the markdown in the database?
    
    # Get conferences based on type
    conferences = CONFERENCE_CATEGORIES[conference_type]
    
    if abstract:
        constraints = [Paper.abstract.contains(x) for x in keywords]
        constraints.append(Paper.year >= start_year)
        constraints.append(Paper.conference.in_(conferences))  # Add conference filter
        with Session() as session:
            papers = session.query(Paper).filter(*constraints).all()
        filter_paper = filter(lambda p: existed_in_tokens(fuzzy_match(p.abstract.lower()), keywords), papers)
    else:
        constraints = [Paper.title.contains(x) for x in keywords]
        constraints.append(Paper.year >= start_year)
        constraints.append(Paper.conference.in_(conferences))  # Add conference filter
        with Session() as session:
            papers = session.query(Paper).filter(*constraints).all()
        #check whether nltk tokenizer data is downloaded
        check_and_download_punkt()
        #tokenize the title and filter out the substring matches
        filter_paper = []
        for paper in papers:
            if all([stemmer.stem(x.lower()) in fuzzy_match(paper.title.lower()) for x in keywords]):
                filter_paper.append(paper)
    # perform customized sorting
    papers = sorted(filter_paper, key=lambda paper: paper.year + conferences.index(paper.conference)/10, reverse=True)
    return papers


def show_papers(papers):
    for paper in papers:
        print(paper)


def main():
    parser = argparse.ArgumentParser(description='Scripts to query the paper database',
                                     usage="%(prog)s [options] -k <keywords>")
    parser.add_argument('-k', type=str, help="keywords to grep, separated by ','. For example, 'linux,kernel,exploit'", default='')
    parser.add_argument('--build-db', action="store_true", help="Builds the database of conference papers")
    parser.add_argument('--abstract', action="store_true", help="Involve abstract into the database's building or query (Need Chrome for building)")
    parser.add_argument('--conference-type', type=str, choices=['security', 'software_engineering', 'all'],
                        default='all', help="Type of conferences to process")
    parser.add_argument('--start-year', type=int, default=2000, 
                        help="Start year for paper search (default: 2000)")
    parser.add_argument('--match-reviewers', nargs=2, metavar=('CFP_URL', 'PDF_PATH'),
                        help="Match potential reviewers for a paper submission. Requires CFP URL and PDF path.")
    parser.add_argument('--sources', type=str, default='dblp,openalex,semanticscholar,website',
                        help="Comma-separated publication sources: dblp,openalex,semanticscholar,website")
    parser.add_argument('--text-source', type=str, choices=['auto', 'pdf', 'abstract', 'title'],
                        default='auto', help="Text source for similarity: auto, pdf, abstract, or title")
    parser.add_argument('--output', type=str, choices=['table', 'json', 'markdown'],
                        default='table', help="Output format for reviewer matches")
    parser.add_argument('--top-n', type=int, default=8,
                        help="Number of reviewers to display (default: 8)")
    parser.add_argument('--top-papers', type=int, default=3,
                        help="Number of similar papers per reviewer (default: 3)")
    parser.add_argument('--min-pdf', type=int, default=0,
                        help="Minimum number of PDFs required to include a reviewer (default: 0)")
    args = parser.parse_args()

    if args.build_db:
        print(f"Building db for {args.conference_type} conferences...")
        build_db(args.abstract, args.conference_type)
    elif args.match_reviewers:
        cfp_url, pdf_path = args.match_reviewers
        print(f"Matching reviewers for submission: {pdf_path}")
        print(f"Using CFP URL: {cfp_url}")
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        ranked_reviewers = match_reviewers(
            cfp_url,
            pdf_path,
            sources=sources,
            text_source=args.text_source,
            output_format=args.output,
            top_n=args.top_n,
            top_papers=args.top_papers,
            min_pdf=args.min_pdf,
        )
        print(f"\nFound {len(ranked_reviewers)} potential reviewers.")
    else:
        assert DB_PATH.exists(), f"need to build a paper database first to perform wanted queries"
        keywords = [x.strip() for x in args.k.split(',')]
        # Remove empty strings from keywords list
        keywords = [k for k in keywords if k]
        
        if keywords:
            logger.info("Grep based on the following keywords: %s", ', '.join(keywords))
        else:
            logger.warning("No keyword is provided. Return all the papers.")

        papers = grep(keywords, args.abstract, args.start_year, args.conference_type)
        logger.debug(f"Found {len(papers)} papers from year {args.start_year} onwards")
        show_papers(papers)


if __name__ == "__main__":
    main()
