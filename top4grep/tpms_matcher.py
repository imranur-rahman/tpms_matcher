# tpms_matcher.py

import os
import requests
import sqlite3
import time
import math
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime
from urllib.parse import urljoin
import json

# ----------- Config ----------- #
DBLP_SEARCH_API = "https://dblp.org/search/publ/api?q=author:{}&format=json"
DBLP_AUTHOR_BASE = "https://dblp.org"
DECAY_HALF_LIFE = 5  # years
MARKITDOWN = MarkItDown()
CURRENT_YEAR = datetime.now().year

# ----------- Utility Functions ----------- #
def exponential_decay_weight(year, current_year=CURRENT_YEAR, half_life=DECAY_HALF_LIFE):
    age = current_year - int(year)
    return math.exp(-math.log(2) * age / half_life)

def fetch_html(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

# ----------- Step 1: Crawl CFP Page ----------- #
def extract_pc_members(cfp_url):
    html = fetch_html(cfp_url)
    soup = BeautifulSoup(html, 'html.parser')

    # Extract detailed member information
    pc_members = []
    
    # Look for list items with class "list-group-item" (specific structure)
    for li in soup.find_all("li", class_="list-group-item"):
        member_info = {}
        
        # Extract profile URL
        link = li.find("a", class_="navigate")
        if link:
            member_info['url'] = link.get("href")
            if member_info['url'] and not member_info['url'].startswith("http"):
                member_info['url'] = urljoin(cfp_url, member_info['url'])
        else:
            member_info['url'] = None
        
        # Extract name and role from first media-heading
        name_headings = li.find_all("h5", class_="media-heading")
        if len(name_headings) >= 1:
            # First heading contains the name and possibly a role
            name_heading = name_headings[0]
            
            # Extract role from pull-right span if it exists
            role_span = name_heading.find("span", class_="pull-right")
            if role_span:
                role_text = role_span.get_text().strip()
                member_info['role'] = role_text if role_text else None
                # Remove the role span to get clean name
                role_span.extract()
            else:
                member_info['role'] = None
            
            # Now get the clean name
            original_name = name_heading.get_text().strip()
            member_info['name'] = original_name
            member_info['decoded_name'] = original_name.encode().decode('unicode_escape')
            
            # Second heading contains university and possibly location
            if len(name_headings) >= 2:
                affiliation_text = name_headings[1].get_text().strip()
                
                # Parse affiliation - could be "University" or "University, Country"
                if ',' in affiliation_text:
                    parts = [part.strip() for part in affiliation_text.split(',')]
                    member_info['university'] = parts[0]
                    member_info['location'] = ', '.join(parts[1:])
                else:
                    member_info['university'] = affiliation_text
                    member_info['location'] = None
            else:
                member_info['university'] = None
                member_info['location'] = None
                
            # Only add if we have a valid name
            if member_info['name'] and len(member_info['name'].split()) >= 2:
                pc_members.append(member_info)
    
    # Fallback to original extraction method if the specific structure isn't found
    if not pc_members:
        for tag in soup.find_all(['li', 'p', 'td']):
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
                    'name': text,
                    'decoded_name': text.encode().decode('unicode_escape'),
                    'url': url,
                    'university': None,
                    'location': None,
                    'role': None
                }
                pc_members.append(member_info)

    return pc_members

# ----------- Step 2: Fetch Publications ----------- #
def fetch_dblp_publications(author_name):
    query = author_name.replace(" ", "+")
    all_papers = []
    hits_per_page = 1000  # Maximum allowed by DBLP
    first_result = 0
    
    print(f"Fetching publications for {author_name}...")
    
    while True:
        # Add pagination parameters
        url = f"{DBLP_SEARCH_API}&h={hits_per_page}&f={first_result}".format(query)
        resp = requests.get(url)
        
        if resp.status_code != 200:
            print(f"Error fetching publications for {author_name}: {resp.status_code}")
            print(resp.text)
            break
            
        data = resp.json()
        result = data.get("result", {})
        hits = result.get("hits", {}).get("hit", [])
        
        if not hits:
            break
            
        # Process current batch
        for hit in hits:
            info = hit.get("info", {})
            year = info.get("year")
            # Handle case where year might be an array
            if isinstance(year, list):
                year = year[0] if year else None
            title = info.get("title")
            print (f"Processing paper: {title} ({year})")
            if year and title:
                all_papers.append((title, int(year)))
        
        # Check if we got all results
        total_hits = int(result.get("hits", {}).get("@total", "0"))
        first_result += len(hits)
        
        print(f"Fetched {first_result}/{total_hits} publications for {author_name}")
        
        # Break if we've fetched all results
        if first_result >= total_hits:
            break
            
        # Add a small delay to be respectful to DBLP
        time.sleep(0.5)
    
    print(f"Found {len(all_papers)} total publications for {author_name}")
    return all_papers

# ----------- Step 3: Build Reviewer Profiles ----------- #
def build_reviewer_profiles(pc_list):
    profiles = {}
    for member_info in pc_list:
        name = member_info['name']
        decoded_name = member_info['decoded_name']
        
        # Try fetching papers with original name first
        papers = fetch_dblp_publications(name)
        
        # If no papers found with original name, try decoded name
        if not papers and name != decoded_name:
            papers = fetch_dblp_publications(decoded_name)
        
        if not papers:
            continue
            
        text_blob = []
        for title, year in papers:
            weight = exponential_decay_weight(year)
            text_blob.append((title, weight))
        profiles[name] = {
            'papers': text_blob,
            'info': member_info
        }
        time.sleep(1)  # Avoid DBLP throttling
    return profiles

# ----------- Step 4: Parse Submission PDF ----------- #
def extract_text_from_pdf(pdf_path):
    result = MARKITDOWN.convert(pdf_path)
    return result.text_content

# ----------- Step 5: Build Corpus and Vectorize ----------- #
def vectorize_profiles_and_paper(profiles, paper_text):
    corpus = []
    reviewer_names = []
    for reviewer, profile_data in profiles.items():
        weighted_titles = profile_data['papers']
        weighted_text = " ".join([title * round(weight * 10) for title, weight in weighted_titles])
        corpus.append(weighted_text)
        reviewer_names.append(reviewer)
    corpus.append(paper_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(corpus)
    paper_vec = vectors[-1]
    reviewer_vecs = vectors[:-1]

    scores = cosine_similarity(paper_vec, reviewer_vecs)[0]
    ranked = []
    for i, (name, score) in enumerate(zip(reviewer_names, scores)):
        member_info = profiles[name]['info']
        ranked.append((name, score, member_info))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# ----------- Main Function ----------- #
def match_reviewers(cfp_url, submission_pdf):
    print("Extracting PC members...")
    pc_list = extract_pc_members(cfp_url)
    print(f"Found {len(pc_list)} PC members")
    # print(json.dumps(pc_list, indent=2))

    print("Building reviewer profiles...")
    profiles = build_reviewer_profiles(pc_list)

    print("Parsing submission PDF...")
    paper_text = extract_text_from_pdf(submission_pdf)

    print("Computing similarity scores...")
    ranked_reviewers = vectorize_profiles_and_paper(profiles, paper_text)

    print("Top matches:")
    for i, (name, score, info) in enumerate(ranked_reviewers[:10], 1):
        print(f"{i}. {name} - Score: {score:.4f}")
        if info['decoded_name'] != info['name']:
            print(f"   Decoded Name: {info['decoded_name']}")
        if info['role']:
            print(f"   Role: {info['role']}")
        if info['university']:
            print(f"   University: {info['university']}")
        if info['location']:
            print(f"   Location: {info['location']}")
        if info['url']:
            print(f"   Profile: {info['url']}")
        print()

    return ranked_reviewers

# Example usage:
# match_reviewers("https://conf.researchr.org/track/ase-2025/ase-2025-papers", "./example_submission.pdf")
