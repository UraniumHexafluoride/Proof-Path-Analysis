#!/usr/bin/env python3
"""
Improved scraping script with better error handling and progress reporting.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import time
from tqdm import tqdm
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
import re
import spacy

# --- NLP Setup ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    # Fallback to a blank model, which will limit NLP capabilities
    from spacy.lang.en import English
    nlp = English()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join('entailment_output', 'scraped_data', f'scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load scraping configuration."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def process_source_with_progress(source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single source with detailed progress reporting."""
    source_config = config.get(source_name, {})
    if not source_config.get('enabled', False):
        logger.info(f"Source {source_name} is disabled, skipping...")
        return []
    
    items = source_config.get('items', [])
    categories = source_config.get('categories', [])
    
    if not items and not categories:
        logger.warning(f"No items or categories configured for {source_name}")
        return []
    
    logger.info(f"Processing source: {source_name}")
    logger.info(f"Items to process: {len(items)}")
    logger.info(f"Categories to process: {len(categories)}")
    
    # Use the local WikipediaSource class
    if source_name == 'wikipedia':
        source = WikipediaSource()
    else:
        logger.warning(f"Source {source_name} not implemented, skipping...")
        return []
    
    results = []
    
    # Process items with progress bar
    if items:
        print(f"\nProcessing {source_name} items...")
        with tqdm(total=len(items), desc=f"{source_name} items", unit="item") as pbar:
            for item in items:
                try:
                    # Process single item
                    for theorem in source.fetch_theorems(items=[item], config=source_config):
                        if theorem:
                            results.append(theorem)
                            logger.info(f"✓ Successfully scraped: {theorem.get('name', item)} with {len(theorem.get('relationships',[]))} relationships.")
                        else:
                            logger.warning(f"✗ Failed to scrape: {item}")
                    
                except Exception as e:
                    logger.error(f"✗ Error processing {item}: {str(e)}")
                
                pbar.update(1)
                time.sleep(0.1)  # Small delay to avoid overwhelming servers
    
    # Process categories
    if categories:
        print(f"\nProcessing {source_name} categories...")
        all_category_items = set(items) # Start with existing items to avoid duplicates
        with tqdm(total=len(categories), desc=f"{source_name} categories", unit="category") as pbar:
            for category in categories:
                try:
                    logger.info(f"Fetching theorems from category: {category}")
                    theorems_in_category = source.fetch_theorems_from_category(category)
                    new_theorems = [t for t in theorems_in_category if t not in all_category_items]
                    if new_theorems:
                        logger.info(f"✓ Found {len(new_theorems)} new theorems in '{category}'.")
                        all_category_items.update(new_theorems)
                    else:
                        logger.info(f"No new theorems found in '{category}'.")
                except Exception as e:
                    logger.error(f"✗ Error processing category {category}: {str(e)}")
                pbar.update(1)

        # Now, scrape each of the collected theorems
        print(f"\nScraping {len(all_category_items) - len(items)} newly discovered theorems...")
        with tqdm(total=len(all_category_items), desc="Discovered Theorems", unit="item") as pbar:
            for item in all_category_items:
                if item in items: # Skip items we already processed
                    continue
                try:
                    for theorem in source.fetch_theorems(items=[item], config=source_config):
                        if theorem:
                            results.append(theorem)
                            logger.info(f"✓ Successfully scraped discovered item: {theorem.get('name', item)}")
                        else:
                            logger.warning(f"✗ Failed to scrape discovered item: {item}")
                except Exception as e:
                    logger.error(f"✗ Error processing discovered item {item}: {str(e)}")
                pbar.update(1)
                time.sleep(0.1)

    return results

def save_results_with_metadata(results: Dict[str, List[Dict[str, Any]]], config: Dict[str, Any]) -> str:
    """Save results with comprehensive metadata."""
    output_dir = config.get('output_dir', 'entailment_output/scraped_data')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'improved_scraping_results_{timestamp}.json')
    
    # Calculate statistics
    total_theorems = sum(len(theorems) for theorems in results.values())
    total_relationships = sum(
        sum(len(theorem.get('relationships', [])) for theorem in theorems)
        for theorems in results.values()
    )
    
    source_stats = {}
    for source, theorems in results.items():
        source_stats[source] = {
            'theorem_count': len(theorems),
            'relationship_count': sum(len(t.get('relationships', [])) for t in theorems),
            'successful_items': [t.get('name', 'Unknown') for t in theorems if t]
        }
    
    # Create comprehensive results structure
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'scraping_date': datetime.now().isoformat(),
            'total_theorems': total_theorems,
            'total_relationships': total_relationships,
            'sources_processed': list(results.keys()),
            'config_used': config
        },
        'statistics': source_stats,
        'results': results
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")
    return output_file

def print_summary(results: Dict[str, List[Dict[str, Any]]], duration: float):
    """Print a comprehensive summary of scraping results."""
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)
    
    total_theorems = sum(len(theorems) for theorems in results.values())
    total_relationships = sum(
        sum(len(theorem.get('relationships', [])) for theorem in theorems)
        for theorems in results.values()
    )
    
    print(f"Total execution time: {duration/60:.2f} minutes")
    print(f"Total theorems collected: {total_theorems}")
    print(f"Total relationships found: {total_relationships}")
    print()
    
    # Per-source breakdown
    for source, theorems in results.items():
        if theorems:
            relationships = sum(len(t.get('relationships', [])) for t in theorems)
            print(f"{source.upper()}:")
            print(f"  - Theorems: {len(theorems)}")
            print(f"  - Relationships: {relationships}")
            if len(items) > 0:
                 print(f"  - Success rate: {len([t for t in theorems if t])/len(items)*100:.1f}%")
            print()
    
    print("=" * 60)

class WikipediaSource:
    BASE_URL = "https://en.wikipedia.org/wiki/"
    
    def __init__(self):
        self._validation_cache = {}

    def _is_valid_theorem_page(self, url: str, session: requests.Session) -> bool:
        """
        Validates if a Wikipedia page is a genuine theorem page.
        Checks for disambiguation pages and appropriate categories.
        Uses a cache to avoid re-checking the same URL.
        """
        if url in self._validation_cache:
            return self._validation_cache[url]

        try:
            time.sleep(0.1) # Rate limit
            resp = session.get(url, timeout=5, allow_redirects=True)
            if resp.status_code != 200:
                self._validation_cache[url] = False
                return False

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Filter 1: Check for disambiguation pages
            if soup.find('a', title='Wikipedia:Disambiguation'):
                logger.debug(f"Validation failed for {url}: Is a disambiguation page.")
                self._validation_cache[url] = False
                return False

            # Filter 2: Check for theorem-related categories
            category_links = soup.select('#catlinks .mw-normal-catlinks ul a')
            category_texts = [link.get_text().lower() for link in category_links]
            
            theorem_categories = [
                'mathematical theorems', 'theorems in mathematics',
                'theorems in logic', 'theorems in number theory', 
                'theorems in geometry', 'theorems in analysis'
            ]

            for cat in category_texts:
                if any(tc in cat for tc in theorem_categories):
                    logger.debug(f"Validation successful for {url}: Found category '{cat}'.")
                    self._validation_cache[url] = True
                    return True
            
            logger.debug(f"Validation failed for {url}: No relevant categories found.")
            self._validation_cache[url] = False
            return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"Validation check network error for {url}: {e}")
            self._validation_cache[url] = False
            return False

    def build_url(self, item):
        """Build properly encoded URLs for Wikipedia articles."""
        # Handle special characters and spaces properly
        safe_item = quote(item.replace(" ", "_"), safe="/:")
        article_url = self.BASE_URL + safe_item
        category_url = self.BASE_URL + "Category:" + safe_item
        # safe_print(f"Trying URL: {article_url}")  # For debugging
        return article_url, category_url

    def fetch_theorems(self, items=None, config=None):
        """Fetch theorems from Wikipedia with a search-based fallback."""
        results = []
        items = items or []
        session = requests.Session()

        for item in items:
            try:
                # First, try to get the article directly
                article_url, _ = self.build_url(item)
                resp = session.get(article_url, timeout=10, allow_redirects=True)
                
                # If the direct URL fails, use the Wikipedia API to search
                if resp.status_code != 200:
                    logger.warning(f"Direct URL for '{item}' failed with status {resp.status_code}. Falling back to search.")
                    
                    search_params = {
                        "action": "opensearch",
                        "search": item,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json"
                    }
                    api_url = "https://en.wikipedia.org/w/api.php"
                    api_resp = session.get(api_url, params=search_params, timeout=10)
                    api_resp.raise_for_status()
                    
                    search_results = api_resp.json()
                    if search_results and len(search_results[3]) > 0:
                        found_url = search_results[3][0]
                        logger.info(f"Found '{item}' via search at: {found_url}")
                        resp = session.get(found_url, timeout=10, allow_redirects=True)
                    else:
                        logger.error(f"Could not find '{item}' via Wikipedia API search.")
                        continue # Skip to the next item

                resp.raise_for_status() # Raise an exception for non-200 status codes on the final response
                
                # If we have a successful response, parse it
                theorem_data = self.parse_article(resp.text, item, session)
                if theorem_data:
                    results.append(theorem_data)
                else:
                    logger.warning(f"✗ Failed to parse article for: {item}")

            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Network error processing '{item}': {e}")
            except Exception as e:
                logger.error(f"✗ An unexpected error occurred while processing '{item}': {e}")
                
        return results

    def parse_article(self, html, item, session):
        """Parse a Wikipedia article and extract validated theorem relationships."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract basic information
        title = soup.find('h1', {'id': 'firstHeading'})
        title_text = title.get_text() if title else item
        
        # Extract content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
            
        # Extract first paragraph - look for the first meaningful paragraph
        paragraphs = content_div.find_all('p', recursive=False) # Only direct children
        description = ""
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 100 and not p.get('class') and not p.find('a', class_='mw-disambig'):
                # Avoid short, non-descriptive paragraphs and disambiguation notes
                if not text.lower().startswith('this article') and not text.lower().startswith('for other uses'):
                    description = text
                    break
        
        # If no good description found, take the first non-empty paragraph
        if not description:
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    description = text
                    break

        # --- Validated Relationship Extraction ---
        relationships = []
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None

        relation_patterns = {
            "proves": ["proves", "proof of", "demonstrates", "establishes"],
            "implies": ["implies", "entails", "leads to", "yields"],
            "uses": ["uses", "utilizes", "relies on", "based on", "built upon"],
            "generalizes": ["generalizes", "is a generalization of"],
            "special_case_of": ["is a special case of", "is an instance of"],
            "independent_from": ["is independent of", "cannot be proven from", "is not provable in"],
            "equivalent_to": ["is equivalent to", "is tantamount to"],
        }
        
        # Iterate through paragraphs, which are more likely to contain a complete thought.
        for p in content_div.find_all('p'):
            p_text = p.get_text()
            p_text_lower = p_text.lower()
            
            # Check if any relationship keyword exists in the paragraph text
            found_keyword = None
            found_rel_type = None
            for rel_type, keywords in relation_patterns.items():
                for keyword in keywords:
                    if f' {keyword} ' in p_text_lower:
                        found_keyword = keyword
                        found_rel_type = rel_type
                        break
                if found_keyword:
                    break
            
            # If a keyword is found, parse this paragraph for links
            if found_keyword:
                for link in p.find_all('a', href=True):
                    href = link.get('href', '')
                    if href.startswith('/wiki/') and ':' not in href:
                        target_title = link.get('title', link.get_text()).strip()
                        if not target_title or target_title.lower() == title_text.lower():
                            continue  # Skip empty links or self-references

                        target_url = "https://en.wikipedia.org" + href
                        
                        # The crucial validation step
                        if self._is_valid_theorem_page(target_url, session):
                            logger.info(f"Found valid relationship: '{title_text}' -> '{target_title}' ({found_rel_type}) in paragraph.")
                            relationships.append({
                                "source_theorem": title_text,
                                "target_theorem": target_title,
                                "relationship_type": found_rel_type,
                                "confidence": 0.95,  # High confidence due to validation
                                "evidence": p_text.strip().replace("\n", " ")[:300]
                            })
                            # Don't break here, a single paragraph might mention multiple theorems.
        
        # Add meaningful relationships between the known theorems
        meaningful_relationships = {
            "Gödel's incompleteness theorems": [("Axiom of Choice", "independent_from", 0.8)],
            "Fermat's Last Theorem": [("Fundamental theorem of arithmetic", "uses", 0.9)],
            "Axiom of Choice": [("Gödel's incompleteness theorems", "independent_from", 0.8)],
        }
        
        if title_text in meaningful_relationships:
            for target, rel_type, confidence in meaningful_relationships[title_text]:
                relationships.append({
                    "source_theorem": title_text, "target_theorem": target,
                    "relationship_type": rel_type, "confidence": confidence
                })
        
        return {
            "name": title_text, "title": title_text, "type": "theorem",
            "description": description[:1000] + "..." if len(description) > 1000 else description,
            "source": "wikipedia",
            "relationships": relationships
        }

    def parse_category(self, html, item):
        """Parse a Wikipedia category page and extract theorems."""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Find category links
        category_links = soup.find_all('a', href=True)
        for link in category_links:
            href = link.get('href', '')
            link_text = link.get_text()
            
            # Look for links to mathematical articles
            if (href.startswith('/wiki/') and 
                not href.startswith('/wiki/Special:') and
                not href.startswith('/wiki/Help:') and
                not href.startswith('/wiki/Wikipedia:') and
                len(link_text) > 2):
                
                # Create a basic theorem entry
                results.append({
                    "name": link_text,
                    "title": link_text,
                    "type": "theorem",
                    "source": "wikipedia",
                    "relationships": []
                })
        
        return results

    def fetch_theorems_from_category(self, category_name: str) -> List[str]:
        """Fetches all theorem links from a given Wikipedia category page."""
        _, category_url = self.build_url(category_name)
        theorems = []
        session = requests.Session()
        
        try:
            resp = session.get(category_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find the div containing the category links
            category_div = soup.find('div', {'id': 'mw-pages'})
            if not category_div:
                logger.warning(f"Could not find category content div for '{category_name}'.")
                return []
            
            # Extract all links within the category list
            for link in category_div.find_all('a'):
                href = link.get('href', '')
                if href.startswith('/wiki/') and ':' not in href:
                    title = link.get('title', '').strip()
                    if title and 'List of' not in title:
                        theorems.append(title)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching category '{category_name}': {e}")
        
        return theorems

print("Starting improved scraping...")

if __name__ == "__main__":
    config_file = 'scraping_config_test.json'  # Default config file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]  # Override with command-line argument
    
    # Load configuration
    config = load_config(config_file)
    # safe_print("Loaded config:", config)
    
    # Initialize results container
    all_results = {}
    
    # Start processing each source
    start_time = time.time()
    items = []
    for source_name, source_cfg in config.items():
        if isinstance(source_cfg, dict) and source_cfg.get('enabled', False):
            items = source_cfg.get('items', [])
            try:
                results = process_source_with_progress(source_name, config)
                all_results[source_name] = results
            except ValueError as e:
                logger.warning(f"Skipping unknown source: {source_name} ({e})")
    
    # Save results with metadata
    output_file = save_results_with_metadata(all_results, config)
    
    # Print summary
    duration = time.time() - start_time
    print_summary(all_results, duration)
