"""
Category-based theorem scraper.
"""

import logging
from bs4 import BeautifulSoup
from typing import Set, Iterator
from urllib.parse import urljoin, quote
import time
import random
from .theorem_mappings import normalize_theorem_url, should_skip_page

class CategoryScraper:
    """Scraper for category-based theorem collection."""
    
    def __init__(self, base_url: str):
        """Initialize the category scraper."""
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.session = self._create_session()
        self.processed_urls = set()
    
    def scrape_category(self, category: str, depth: int = 1) -> Iterator[str]:
        """
        Scrape theorems from a category.
        
        Args:
            category: Category name to scrape
            depth: How deep to follow subcategories (default: 1)
            
        Returns:
            Iterator of theorem URLs
        """
        try:
            # Normalize category name
            normalized_category = normalize_theorem_url(category)
            
            # Construct category URL
            category_url = f"{self.base_url}/wiki/Category:{quote(normalized_category)}"
            
            # Try to fetch the page
            response = self._make_request(category_url)
            
            # If normalized URL fails, try original
            if response.status_code == 404:
                category_url = f"{self.base_url}/wiki/Category:{quote(category)}"
                response = self._make_request(category_url)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch category {category}")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Process pages in this category
            for link in soup.select('div.mw-category a'):
                if not link.get('href'):
                    continue
                    
                url = urljoin(self.base_url, link.get('href'))
                
                # Skip if we've seen this URL
                if url in self.processed_urls:
                    continue
                    
                self.processed_urls.add(url)
                
                # Skip navigation/metadata pages
                page_name = url.split('/')[-1]
                if should_skip_page(page_name):
                    continue
                
                # If it's a theorem page, yield it
                if not url.startswith(f"{self.base_url}/wiki/Category:"):
                    yield url
                # If it's a subcategory and we have depth remaining, process it
                elif depth > 0:
                    subcategory = url.split(':')[-1]
                    yield from self.scrape_category(subcategory, depth - 1)
                
                # Add random delay
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            self.logger.error(f"Error scraping category {category}: {str(e)}")
    
    def _make_request(self, url: str, max_retries: int = 3):
        """Make an HTTP request with retries."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                if response.status_code == 404:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {response.status_code} {response.reason} for url: {url}")
                        time.sleep(random.uniform(1, 3))
                        continue
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(random.uniform(1, 3))
                    continue
                raise
        return None
    
    def _create_session(self):
        """Create a session with appropriate headers."""
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session 