"""
Scraper for Wolfram MathWorld mathematical content.
"""

import requests
from bs4 import BeautifulSoup
from typing import Generator, Dict, Any, List, Optional
from .base_source import MathSource, ScrapedTheorem
import logging
import re
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class MathWorldSource(MathSource):
    """Scraper for Wolfram MathWorld encyclopedia."""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://mathworld.wolfram.com"
        self.theorem_categories = [
            "/topics/Theorems.html",
            "/topics/FundamentalTheorems.html",
            "/topics/LogicandFoundations.html",
            "/topics/NumberTheory.html",
            "/topics/Algebra.html",
            "/topics/Analysis.html"
        ]
    
    def _extract_theorem_info(self, content: BeautifulSoup) -> dict:
        """Extract theorem information from page content."""
        info = {
            'related_topics': [],
            'references': [],
            'formulas': [],
            'classifications': []
        }
        
        # Extract related topics
        related = content.find('div', {'class': 'related'})
        if related:
            info['related_topics'] = [a.text for a in related.find_all('a')]
        
        # Extract references
        refs = content.find('div', {'class': 'references'})
        if refs:
            info['references'] = [ref.text for ref in refs.find_all('li')]
        
        # Extract mathematical formulas
        formulas = content.find_all('img', {'class': 'math'})
        if formulas:
            info['formulas'] = [img.get('alt', '') for img in formulas]
        
        # Extract classifications
        classifications = content.find('div', {'class': 'classifications'})
        if classifications:
            info['classifications'] = [cls.text for cls in classifications.find_all('a')]
        
        return info
    
    def fetch_theorems(self, items: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch theorems from MathWorld.
        
        Args:
            items: Optional list of specific categories to fetch (e.g., 'FundamentalTheorems')
            
        Yields:
            Dictionary containing theorem data
        """
        categories = items if items else self.theorem_categories
        
        for category in categories:
            try:
                # Add .html extension if not present
                if not category.endswith('.html'):
                    category = f"{category}.html"
                
                # Add /topics/ prefix if not present
                if not category.startswith('/topics/'):
                    category = f"/topics/{category}"
                
                url = urljoin(self.base_url, category)
                page = self.session.get(url)
                page.raise_for_status()
                soup = BeautifulSoup(page.text, 'html.parser')
                
                # Find theorem links
                for link in soup.find_all('a', href=re.compile(r'/\w+Theorem\.html')):
                    try:
                        theorem_url = urljoin(self.base_url, link['href'])
                        theorem_page = self.session.get(theorem_url)
                        theorem_page.raise_for_status()
                        theorem_soup = BeautifulSoup(theorem_page.text, 'html.parser')
                        
                        # Extract theorem content
                        title = theorem_soup.find('h1').text if theorem_soup.find('h1') else link.text
                        content = theorem_soup.find('div', {'class': 'content'})
                        description = content.find('p').text if content and content.find('p') else ""
                        
                        # Extract additional information
                        info = self._extract_theorem_info(theorem_soup)
                        
                        metadata = {
                            'source_type': 'mathworld',
                            'url': theorem_url,
                            'related_topics': info['related_topics'],
                            'references': info['references'],
                            'formulas': info['formulas'],
                            'classifications': info['classifications']
                        }
                        
                        yield {
                            'name': title,
                            'description': description,
                            'source_url': theorem_url,
                            'metadata': metadata
                        }
                        
                        self._random_delay()
                        
                    except Exception as e:
                        logger.error(f"Error processing theorem {link.text}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing category {category}: {str(e)}")
                continue 