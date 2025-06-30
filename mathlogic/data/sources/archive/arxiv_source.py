"""
arXiv theorem scraper.
"""

import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, Iterator, List, Optional
from urllib.parse import urljoin
from .base_source import MathSource
import re

class ArxivSource(MathSource):
    """Scraper for arXiv."""
    
    def __init__(self):
        """Initialize the arXiv scraper."""
        super().__init__()
        self.base_url = "https://arxiv.org"
        self.logger = logging.getLogger(__name__)
    
    def fetch_theorems(self, items: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Fetch theorems from arXiv.
        
        Args:
            items: Optional list of specific categories to fetch (e.g., 'math.LO')
            
        Returns:
            Iterator of theorem dictionaries
        """
        try:
            # Default categories if none specified
            categories = items if items else ['math.LO', 'math.AG', 'math.NT']
            
            for category in categories:
                try:
                    # Get papers from category
                    url = f"{self.base_url}/list/{category}/recent"
                    response = self._make_request(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Process each paper - try both old and new selectors
                    papers = soup.select('dt.arxiv-result, div.list-title')
                    
                    for paper in papers:
                        try:
                            # Get paper title - handle both old and new layouts
                            title_elem = paper.find('div', class_='title') or paper.find('a')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            if not title or not self._is_theorem_title(title):
                                continue
                            
                            # Get paper details - handle both old and new layouts
                            paper_link = paper.find('a', href=re.compile(r'/abs/\d+\.\d+'))
                            if not paper_link:
                                continue
                                
                            paper_id = paper_link['href'].split('/')[-1]
                            paper_url = f"{self.base_url}/abs/{paper_id}"
                            
                            # Fetch paper details
                            paper_response = self._make_request(paper_url)
                            paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
                            
                            # Extract data
                            description = self._extract_description(paper_soup)
                            if not description:
                                continue
                            
                            metadata = self._extract_metadata(paper_soup)
                            relationships = self._extract_relationships(paper_soup, title)
                            
                            yield {
                                'name': title,
                                'description': description,
                                'source_url': paper_url,
                                'metadata': metadata,
                                'relationships': relationships
                            }
                            
                            self._random_delay()
                            
                        except Exception as e:
                            self.logger.error(f"Error processing paper {title if 'title' in locals() else 'unknown'}: {str(e)}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Error processing category {category}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error fetching from arXiv: {str(e)}")
    
    def _is_theorem_title(self, title: str) -> bool:
        """Check if a paper title likely contains a theorem."""
        theorem_keywords = [
            'theorem', 'lemma', 'proposition', 'conjecture',
            'proof', 'corollary', 'axiom'
        ]
        return any(keyword in title.lower() for keyword in theorem_keywords)
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract paper abstract as theorem description."""
        try:
            # Try both old and new abstract selectors
            abstract = soup.select_one('blockquote.abstract, div.abstract')
            if abstract:
                return abstract.text.strip()
            return ""
        except Exception as e:
            self.logger.warning(f"Error extracting description: {str(e)}")
            return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract paper metadata."""
        try:
            metadata = {
                'source': 'arXiv',
                'authors': [],
                'categories': [],
                'date': None
            }
            
            # Extract authors - handle both old and new layouts
            authors = soup.select('div.authors a, div.authors span.a')
            metadata['authors'] = [author.text.strip() for author in authors]
            
            # Extract categories - handle both old and new layouts
            categories = soup.select('span.primary-subject, span.subject-class')
            metadata['categories'] = [cat.text.strip() for cat in categories]
            
            # Extract date - handle both old and new layouts
            date = soup.select_one('div.submission-history, div.dateline')
            if date:
                metadata['date'] = date.text.strip().split('\n')[0]
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}")
            return {'source': 'arXiv'}
    
    def _extract_relationships(self, soup: BeautifulSoup, title: str) -> List[Dict[str, Any]]:
        """Extract paper relationships."""
        relationships = []
        
        try:
            # Extract references from abstract
            abstract = soup.select_one('blockquote.abstract, div.abstract')
            if abstract:
                text = abstract.text.lower()
                
                # Look for references to other theorems
                theorem_pattern = r'([\w\s]+)(?:theorem|lemma|proposition)'
                matches = re.finditer(theorem_pattern, text)
                
                for match in matches:
                    theorem_name = match.group(1).strip()
                    if theorem_name and theorem_name != title.lower():
                        relationships.append({
                            'source_theorem': title,
                            'target_theorem': theorem_name,
                            'relationship_type': 'references',
                            'confidence': 0.6
                        })
            
        except Exception as e:
            self.logger.warning(f"Error extracting relationships: {str(e)}")
        
        return relationships