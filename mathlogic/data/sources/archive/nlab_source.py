"""
nLab theorem scraper.
"""

import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, Iterator, List, Optional
from .base_source import MathSource

class NLabSource(MathSource):
    """Scraper for nLab."""
    
    def __init__(self):
        """Initialize the nLab scraper."""
        super().__init__()
        self.base_url = "https://ncatlab.org/nlab"  # Changed base URL
        self.logger = logging.getLogger(__name__)
    
    def fetch_theorems(self, items: Optional[List[str]] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Fetch theorems from nLab.
        
        Args:
            items: Optional list of specific theorem names to fetch
            config: Optional configuration dictionary with additional settings
            
        Returns:
            Iterator of theorem dictionaries
        """
        try:
            # Get configuration
            rate_limit = config.get('rate_limit', 3.0) if config else 3.0
            
            # Use items from config if not provided directly
            theorems = items or (config.get('items', []) if config else [])
            
            # Add theorems from categories if present
            if config and 'categories' in config:
                for category in config['categories']:
                    try:
                        # Format category URL - nLab uses + for spaces
                        category_name = category.lower().replace(' ', '+')
                        category_url = f"{self.base_url}/show/{category_name}"

                        response = self._make_request(category_url)
                        if response.status_code == 404:
                            # Try alternate URL formats
                            alternate_formats = [
                                category.replace(' ', '+'),  # Original case with +
                                category.replace(' ', '_'),  # Underscores
                                category.lower().replace(' ', '_'),  # Lowercase with underscores
                                f"{self.base_url}/list/{category_name}",  # List endpoint
                                f"{self.base_url}/all/{category_name}"   # All endpoint
                            ]

                            for alt_format in alternate_formats:
                                if alt_format.startswith('http'):
                                    category_url = alt_format
                                else:
                                    category_url = f"{self.base_url}/show/{alt_format}"
                                response = self._make_request(category_url)
                                if response.status_code == 200:
                                    break
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find theorem links in category
                        for link in soup.find_all('a'):
                            if link.text and any(term in link.text.lower() for term in ['theorem', 'lemma', 'conjecture']):
                                theorems.append(link.text.strip())
                                
                    except Exception as e:
                        self.logger.error(f"Error processing category {category}: {str(e)}")
                        continue
            
            # Remove duplicates while preserving order
            theorems = list(dict.fromkeys(theorems))
            
            for theorem in theorems:
                try:
                    # Format theorem URL - nLab uses + for spaces
                    theorem_name = theorem.lower().replace(' ', '+')
                    theorem_url = f"{self.base_url}/show/{theorem_name}"

                    response = self._make_request(theorem_url)
                    if response.status_code == 404:
                        # Try multiple alternate URL formats
                        alternate_formats = [
                            theorem.replace(' ', '+'),  # Original case with +
                            theorem.replace(' ', '_'),  # Underscores
                            theorem.lower().replace(' ', '_'),  # Lowercase with underscores
                            theorem.replace(' ', '+').replace('_', '+'),  # Ensure all +
                            theorem.replace('theorem', 'Theorem'),  # Capitalize theorem
                            theorem.replace('Theorem', 'theorem'),  # Lowercase theorem
                            # Additional nLab-specific formats
                            theorem.replace(' ', '%20'),  # URL encoding
                            theorem.replace(' ', '-'),  # Hyphens
                            theorem.lower().replace(' ', '-'),  # Lowercase with hyphens
                        ]

                        for alt_name in alternate_formats:
                            theorem_url = f"{self.base_url}/show/{alt_name}"
                            response = self._make_request(theorem_url)
                            if response.status_code == 200:
                                break

                        # Try other endpoints if still not found
                        if response.status_code == 404:
                            for endpoint in ['page', 'entry']:
                                for alt_name in [theorem_name] + alternate_formats[:3]:  # Try top 3 formats
                                    theorem_url = f"{self.base_url}/{endpoint}/{alt_name}"
                                    response = self._make_request(theorem_url)
                                    if response.status_code == 200:
                                        break
                                if response.status_code == 200:
                                    break

                    if response.status_code == 404:
                        self.logger.warning(f"Theorem not found after trying multiple formats: {theorem}")
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract theorem data
                    description = self._extract_description(soup)
                    if not description:
                        continue
                    
                    metadata = self._extract_metadata(soup)
                    relationships = self._extract_relationships(soup, theorem)
                    
                    yield {
                        'name': theorem,
                        'description': description,
                        'source_url': theorem_url,
                        'metadata': metadata,
                        'relationships': relationships
                    }
                    
                    # Use rate limit from config
                    self._random_delay(rate_limit)
                    
                except Exception as e:
                    self.logger.error(f"Error processing theorem {theorem}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error fetching from nLab: {str(e)}")
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract theorem description from the page."""
        try:
            # Find the theorem statement section
            content = soup.select_one('div#revision, div#content')
            if content:
                # Look for theorem statement markers
                for h in content.find_all(['h1', 'h2', 'h3', 'h4']):
                    if any(word in h.text.lower() for word in ['statement', 'definition', 'theorem']):
                        section = h.find_next_sibling()
                        if section:
                            return section.text.strip()
                
                # Fallback to first non-empty paragraph
                for p in content.find_all('p', recursive=False):
                    if p.text.strip():
                        return p.text.strip()
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Error extracting description: {str(e)}")
            return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the theorem page."""
        try:
            metadata = {
                'source': 'nLab',
                'categories': [],
                'related_concepts': [],
                'references': []
            }
            
            # Extract categories from the sidebar
            sidebar = soup.select_one('div#sidebar')
            if sidebar:
                for link in sidebar.find_all('a'):
                    if link.text:
                        metadata['categories'].append(link.text.strip())
            
            # Extract related concepts
            content = soup.select_one('div#revision')
            if content:
                for link in content.find_all('a'):
                    if link.text and not link.text.startswith(('http', 'www')):
                        metadata['related_concepts'].append(link.text.strip())
            
            # Extract references
            references = soup.find('h2', string='References')
            if references:
                section = references.find_next_sibling()
                if section:
                    for ref in section.find_all(['p', 'li']):
                        if ref.text.strip():
                            metadata['references'].append(ref.text.strip())
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}")
            return {'source': 'nLab'}
    
    def _extract_relationships(self, soup: BeautifulSoup, theorem_name: str) -> List[Dict[str, Any]]:
        """Extract relationships from the theorem page."""
        relationships = []
        
        try:
            content = soup.select_one('div#revision')
            if content:
                # Look for theorem names in links
                for link in content.find_all('a'):
                    if link.text and 'theorem' in link.text.lower():
                        related_theorem = link.text.strip()
                        if related_theorem != theorem_name:
                            # Basic relationship based on link existence
                            relationships.append({
                                'source_theorem': theorem_name,
                                'target_theorem': related_theorem,
                                'relationship_type': 'related_to',
                                'confidence': 0.5
                            })
                
                # Look for stronger relationships in text
                text = content.text.lower()
                proof_indicators = ['proves', 'implies', 'shows that', 'demonstrates']
                dependency_indicators = ['depends on', 'requires', 'follows from', 'based on']
                
                for link in content.find_all('a'):
                    if not link.text or not 'theorem' in link.text.lower():
                        continue
                        
                    related_theorem = link.text.strip()
                    if related_theorem == theorem_name:
                        continue
                    
                    # Check context around the link
                    context_start = max(0, text.find(link.text.lower()) - 50)
                    context_end = min(len(text), text.find(link.text.lower()) + len(link.text) + 50)
                    context = text[context_start:context_end]
                    
                    if any(indicator in context for indicator in proof_indicators):
                        relationships.append({
                            'source_theorem': related_theorem,
                            'target_theorem': theorem_name,
                            'relationship_type': 'proves',
                            'confidence': 0.8
                        })
                    elif any(indicator in context for indicator in dependency_indicators):
                        relationships.append({
                            'source_theorem': theorem_name,
                            'target_theorem': related_theorem,
                            'relationship_type': 'depends_on',
                            'confidence': 0.7
                        })
            
        except Exception as e:
            self.logger.warning(f"Error extracting relationships: {str(e)}")
        
        return relationships