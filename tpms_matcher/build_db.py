from datetime import datetime
from pathlib import Path

import requests
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from bs4 import BeautifulSoup

from .utils import new_logger
from .db import Base, Paper
from .abstract import Abstracts

logger = new_logger("DB")
logger.setLevel('WARNING')

# Conference categories
SECURITY_CONFERENCES = ["NDSS", "IEEE S&P", "USENIX", "CCS"]
SE_CONFERENCES = ["ICSE", "FSE", "ASE", "ISSTA", "TSE"]

# Conference mapping dictionary
CONFERENCE_CATEGORIES = {
    "security": SECURITY_CONFERENCES,
    "software_engineering": SE_CONFERENCES,
    "all": SECURITY_CONFERENCES + SE_CONFERENCES
}

# Update conference mappings
NAME_TO_CONF = {
    # Security conferences
    "NDSS": "ndss",
    "IEEE S&P": "sp",
    "USENIX": "uss",
    "CCS": "ccs",
    # Software Engineering conferences and journals
    "ICSE": "icse",
    "FSE": "fse",
    "ASE": "ase",
    "ISSTA": "issta",
    "TSE": "tse"  # IEEE Transactions on Software Engineering
}

NAME_TO_ORG = {
    # Security conferences
    "NDSS": "ndss",
    "IEEE S&P": "sp",
    "USENIX": "uss",
    "CCS": "ccs",
    # Software Engineering conferences and journals
    "ICSE": "icse",
    "FSE": "sigsoft",
    "ASE": "kbse",
    "ISSTA": "issta",
    "TSE": "tse"
}

# Add journal information
JOURNALS = {"TSE"}

PACKAGE_DIR = Path(__file__).resolve().parent
DB_PATH = PACKAGE_DIR / "data" / "papers.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = sqlalchemy.create_engine(f'sqlite:///{str(DB_PATH)}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def save_paper(conf, year, title, authors, abstract):
    logger.debug(f'Adding paper {title} with abstract {abstract[:20]}...')
    session = Session()
    paper = Paper(conference=conf, year=year, title=title, authors=", ".join(authors), abstract=abstract)
    session.add(paper)
    session.commit()
    session.close()

def paper_exist(conf, year, title, authors, abstract):
    session = Session()
    paper = session.query(Paper).filter(Paper.conference==conf, Paper.year==year, Paper.title==title, Paper.abstract==abstract).first()
    session.close()
    return paper is not None

def get_papers(name, year, build_abstract):
    cnt = 0
    org = NAME_TO_ORG[name]
    conf = NAME_TO_CONF[name]

    if build_abstract and name == "NDSS" and (year == 2018 or year == 2016):
        logger.warning(f"Skipping the abstract for NDSS {year} because the website does not contain abstracts.")
        extract_abstract = False
    else:
        extract_abstract = build_abstract
    try:
        if name in JOURNALS:
            # Calculate volume number for TSE based on year
            # 2000 -> volume 26, 2025 -> volume 51
            volume_no = year - 1974  # 2000 - 1974 = 26
            url = f"https://dblp.org/db/journals/{org}/{conf}{volume_no}.html"
        else:
            url = f"https://dblp.org/db/conf/{org}/{conf}{year}.html"

        r = requests.get(url)
        assert r.status_code == 200

        html = BeautifulSoup(r.text, 'html.parser')
        # Update to handle both conference and journal papers
        paper_htmls = html.find_all("li", {'class': ["inproceedings", "article"]})
        for paper_html in paper_htmls:
            title = paper_html.find('span', {'class': 'title'}).text
            authors = [x.text for x in paper_html.find_all('span', {'itemprop': 'author'})]
            if extract_abstract:
                abstract = Abstracts[name].get_abstract(paper_html, title, authors)
            else:
                abstract = ''
            # insert the entry only if the paper does not exist
            if not paper_exist(name, year, title, authors, abstract):
                save_paper(name, year, title, authors, abstract)
            cnt += 1
    except Exception as e:
        logger.warning(f"Failed to obtain papers at {name}-{year}")

    logger.debug(f"Found {cnt} papers at {name}-{year}...")


def build_db(build_abstract, conference_type="all"):
    """
    Build database for selected conference type
    Args:
        build_abstract (bool): Whether to build abstracts
        conference_type (str): Type of conferences to process ('security', 'software_engineering', or 'all')
    """
    if conference_type not in CONFERENCE_CATEGORIES:
        raise ValueError(f"Invalid conference type. Choose from: {list(CONFERENCE_CATEGORIES.keys())}")
    
    conferences = CONFERENCE_CATEGORIES[conference_type]
    for conf in conferences:
        for year in range(2000, datetime.now().year+1):
            get_papers(conf, year, build_abstract)
