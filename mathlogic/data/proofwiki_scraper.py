import requests
from bs4 import BeautifulSoup
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import re
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://proofwiki.org/wiki/"
OUTPUT_DIR = "entailment_output/scraped_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# To avoid overwhelming the server and for ethical scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
RATE_LIMIT_DELAY = 1  # seconds between requests

# Relationship keywords to look for in text
RELATIONSHIP_PATTERNS = {
    'proves': [r'proves', r'proof of', r'proven by', r'demonstrated by'],
    'implies': [r'implies', r'leads to', r'results in', r'consequently'],
    'equivalent': [r'equivalent to', r'if and only if', r'iff', r'equivalent with'],
    'independent': [r'independent of', r'not provable from', r'cannot be proved'],
    'related': [r'related to', r'connected to', r'associated with'],
    'uses': [r'uses', r'requires', r'depends on', r'based on']
}

def fetch_page_content(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetches the HTML content of a given URL with retries."""
    for attempt in range(max_retries):
        try:
            print(f"Fetching: {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            time.sleep(RATE_LIMIT_DELAY)
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Error fetching {url} after {max_retries} attempts: {e}")
                return None
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))  # Exponential backoff
    return None

def extract_field(soup: BeautifulSoup) -> str:
    """Extract the mathematical field from page categories."""
    categories = soup.find_all('div', {'class': 'mw-normal-catlinks'})
    fields = {
        'Set Theory': ['set theory', 'set-theoretic'],
        'Number Theory': ['number theory', 'arithmetic'],
        'Analysis': ['analysis', 'calculus', 'topology'],
        'Logic': ['logic', 'proof theory', 'model theory'],
        'Algebra': ['algebra', 'group theory', 'ring theory'],
        'Geometry': ['geometry', 'geometric'],
        'Combinatorics': ['combinatorics', 'combinatorial'],
        'Probability': ['probability', 'statistics']
    }
    
    if categories:
        cat_text = categories[0].get_text().lower()
        for field, keywords in fields.items():
            if any(keyword in cat_text for keyword in keywords):
                return field
    return "Unknown"

def extract_relationships(soup: BeautifulSoup, title: str) -> List[Dict[str, str]]:
    """Extract relationships from the page content using pattern matching."""
    relationships = []
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        return relationships

    # Get text blocks (paragraphs and lists)
    text_blocks = content.find_all(['p', 'li'])
    
    for block in text_blocks:
        text = block.get_text()
        links = block.find_all('a', href=True)
        
        # For each link, look for relationship patterns in surrounding text
        for link in links:
            href = link['href']
            if not href.startswith('/wiki/') or ':' in href:
                continue
                
            target = href.replace('/wiki/', '').replace('_', ' ')
            if target == title:  # Skip self-references
                continue
            
            # Look for relationship patterns in the text around the link
            text_before = text[:text.find(link.text)].lower()
            text_after = text[text.find(link.text) + len(link.text):].lower()
            context = text_before[-100:] + text_after[:100]  # Look at 100 chars before and after
            
            relation_type = 'mentions'  # Default relationship
            for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
                if any(re.search(pattern, context) for pattern in patterns):
                    relation_type = rel_type
                    break
            
            relationships.append({
                'source': title,
                'target': target,
                'relation_type': relation_type,
                'context': context.strip()
            })
    
    return relationships

def parse_theorem_page(html_content: str, url: str) -> Dict[str, Any]:
    """Parse a ProofWiki theorem page to extract information."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title_tag = soup.find('h1', {'id': 'firstHeading'})
    title = title_tag.text.strip() if title_tag else "Unknown Title"
    
    # Extract description
    description = ""
    first_paragraph = soup.find('div', {'class': 'mw-parser-output'})
    if first_paragraph and first_paragraph.find('p'):
        description = first_paragraph.find('p').get_text(separator=' ', strip=True)
        if len(description) > 500:
            description = description[:497] + "..."
    
    # Extract relationships
    relationships = extract_relationships(soup, title)
    
    # Extract field
    field = extract_field(soup)
    
    # Extract complexity (based on proof length/structure if available)
    proof_section = soup.find('span', {'id': 'Proof'})
    if proof_section and proof_section.parent:
        proof_text = proof_section.parent.find_next('p')
        if proof_text:
            proof_length = len(proof_text.get_text())
            complexity = 'high' if proof_length > 1000 else 'medium' if proof_length > 500 else 'low'
        else:
            complexity = 'unknown'
    else:
        complexity = 'unknown'
    
    # Determine if it's a theorem or definition
    page_type = 'theorem'
    if any(cat.get_text().lower().find('definition') != -1 
           for cat in soup.find_all('div', {'class': 'mw-normal-catlinks'})):
        page_type = 'definition'
    
    metadata = {
        'url': url,
        'scraped_at': time.time(),
        'description': description,
        'field': field,
        'complexity': complexity
    }
    
    return {
        'title': title,
        'type': page_type,
        'relationships': relationships,
        'metadata': metadata
    }

def scrape_proofwiki(start_page_titles: List[str], max_pages: int = 50) -> List[Dict[str, Any]]:
    """Main function to scrape ProofWiki using breadth-first search."""
    scraped_data = []
    visited_urls = set()
    queue = [(title, BASE_URL + title.replace(' ', '_')) 
             for title in start_page_titles]
    
    pages_scraped = 0
    failed_pages = []
    
    while queue and pages_scraped < max_pages:
        title, current_url = queue.pop(0)
        if current_url in visited_urls:
            continue
            
        html_content = fetch_page_content(current_url)
        if html_content:
            try:
                page_data = parse_theorem_page(html_content, current_url)
                scraped_data.append(page_data)
                visited_urls.add(current_url)
                pages_scraped += 1
                
                # Add new links to queue
                for rel in page_data['relationships']:
                    target_title = rel['target']
                    new_url = BASE_URL + target_title.replace(' ', '_')
                    if new_url not in visited_urls and (target_title, new_url) not in queue:
                        queue.append((target_title, new_url))
                
                print(f"Successfully scraped: {title}")
            except Exception as e:
                print(f"Error parsing {current_url}: {e}")
                failed_pages.append((current_url, str(e)))
        else:
            failed_pages.append((current_url, "Failed to fetch content"))
    
    # Save failed pages for later analysis
    if failed_pages:
        with open(os.path.join(OUTPUT_DIR, "failed_pages.json"), 'w') as f:
            json.dump(failed_pages, f, indent=2)
    
    print(f"\nScraping complete:")
    print(f"- Successfully scraped: {pages_scraped} pages")
    print(f"- Failed pages: {len(failed_pages)}")
    return scraped_data

if __name__ == "__main__":
    # Example usage with expanded initial theorems
    initial_theorems = [
        "Gödel's_Incompleteness_Theorems",
        "Continuum_Hypothesis",
        "Axiom_of_Choice",
        "Fermat's_Last_Theorem",
        "Four_Color_Theorem",
        "Riemann_Hypothesis",
        "Zermelo–Fraenkel_Set_Theory",
        "Peano_Axioms",
        "P_versus_NP",
        "Fundamental_Theorem_of_Arithmetic",
        "Fundamental_Theorem_of_Algebra",
        "Completeness_Theorem",
        "Compactness_Theorem",
        "Independence_of_Continuum_Hypothesis"
    ]
    
    # Scrape with increased page limit
    scraped_results = scrape_proofwiki(initial_theorems, max_pages=100)
    
    # Save results
    if scraped_results:
        save_path = os.path.join(OUTPUT_DIR, "proofwiki_data.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({"scraped_pages": scraped_results}, f, indent=2)
        print(f"Data saved to: {save_path}")
    else:
        print("No data scraped.")