"""
Wikipedia theorem scraper.
"""

import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, Iterator, List, Optional
from urllib.parse import quote, urljoin
from .base_source import MathSource
import re

class WikipediaSource(MathSource):
    """Scraper for Wikipedia."""
    
    def __init__(self):
        """Initialize the Wikipedia scraper."""
        super().__init__()
        self.base_url = "https://en.wikipedia.org/wiki"
        self.logger = logging.getLogger(__name__)
    
    def fetch_theorems(self, items: Optional[List[str]] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Fetch theorems from Wikipedia.
        
        Args:
            items: Optional list of specific categories or theorem names to fetch
            config: Optional configuration dictionary with additional settings
            
        Returns:
            Iterator of theorem dictionaries
        """
        try:
            # Get configuration
            depth = config.get('depth', 2) if config else 2
            rate_limit = config.get('rate_limit', 1.5) if config else 1.5
            
            # Use categories from config if provided, otherwise use items
            categories = []
            if config and 'categories' in config:
                categories.extend(config['categories'])
            if items:
                categories.extend(items)
            
            # Add direct theorem pages from config
            theorem_pages = []
            if config and 'items' in config:
                theorem_pages.extend(config['items'])
            
            for item in categories + theorem_pages:
                try:
                    # Handle both category pages and direct theorem pages
                    if item.startswith("Category:"):
                        # Get list of theorems from category page
                        category_name = item[9:]  # Remove "Category:" prefix
                        category_url = f"https://en.wikipedia.org/wiki/Category:{quote(category_name)}"
                        
                        response = self._make_request(category_url)
                        if response.status_code == 404:
                            # Try alternate URL format
                            category_url = f"https://en.wikipedia.org/wiki/Category:{quote(category_name.replace('_', ' '))}"
                            response = self._make_request(category_url)
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find all theorem links in the category
                        for link in soup.select('div.mw-category a, div#mw-pages a'):
                            try:
                                if not link.get('href'):
                                    continue
                                
                                # Skip non-theorem pages
                                if not any(term in link.text.lower() for term in ['theorem', 'lemma', 'conjecture']):
                                    continue
                                
                                theorem_url = urljoin("https://en.wikipedia.org", link['href'])
                                theorem_response = self._make_request(theorem_url)
                                theorem_soup = BeautifulSoup(theorem_response.text, 'html.parser')
                                
                                # Extract theorem data
                                name = link.text.strip()
                                description = self._extract_description(theorem_soup)
                                if not description:
                                    continue
                                
                                metadata = self._extract_metadata(theorem_soup)
                                relationships = self._extract_relationships(theorem_soup, name)
                                
                                yield {
                                    'name': name,
                                    'description': description,
                                    'source_url': theorem_url,
                                    'metadata': metadata,
                                    'relationships': relationships
                                }
                                
                                self._random_delay(rate_limit)
                                
                            except Exception as e:
                                self.logger.error(f"Error processing theorem {link.text if link else 'unknown'}: {str(e)}")
                                continue
                    else:
                        # Direct theorem page
                        theorem_name = item.replace(' ', '_')
                        theorem_url = f"{self.base_url}/{quote(theorem_name)}"

                        theorem_response = self._make_request(theorem_url)
                        if theorem_response.status_code == 404:
                            # Try multiple alternate URL formats
                            alternate_formats = [
                                theorem_name.replace('_', ' '),  # Spaces instead of underscores
                                theorem_name.lower(),  # Lowercase
                                theorem_name.replace('_', '-'),  # Hyphens instead of underscores
                                theorem_name.replace('Theorem', 'theorem'),  # Lowercase theorem
                                theorem_name.replace('_of_', '_'),  # Remove "of"
                                self._get_wikipedia_redirect(theorem_name)  # Check for redirects
                            ]

                            for alt_name in alternate_formats:
                                if alt_name:  # Skip None values
                                    theorem_url = f"{self.base_url}/{quote(alt_name)}"
                                    theorem_response = self._make_request(theorem_url)
                                    if theorem_response.status_code == 200:
                                        break

                            # If still not found, try searching Wikipedia
                            if theorem_response.status_code == 404:
                                search_result = self._search_wikipedia(item)
                                if search_result:
                                    theorem_url = search_result
                                    theorem_response = self._make_request(theorem_url)
                        
                        theorem_soup = BeautifulSoup(theorem_response.text, 'html.parser')
                        
                        # Extract theorem data
                        name = item.replace('_', ' ')
                        description = self._extract_description(theorem_soup)
                        if not description:
                            continue
                        
                        metadata = self._extract_metadata(theorem_soup)
                        relationships = self._extract_relationships(theorem_soup, name)
                        
                        yield {
                            'name': name,
                            'description': description,
                            'source_url': theorem_url,
                            'metadata': metadata,
                            'relationships': relationships
                        }
                        
                        self._random_delay(rate_limit)
                        
                except Exception as e:
                    self.logger.error(f"Error processing category/theorem {item}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error fetching from Wikipedia: {str(e)}")
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract theorem description from the page."""
        try:
            # Find the first paragraph after the lead section
            content = soup.select_one('div.mw-parser-output')
            if content:
                # Skip disambiguation notices and other notices
                for p in content.find_all('p', recursive=False):
                    if not p.find_parent(class_=['hatnote', 'ambox']):
                        text = p.text.strip()
                        if text:
                            return text
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Error extracting description: {str(e)}")
            return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the theorem page."""
        try:
            metadata = {
                'source': 'Wikipedia',
                'categories': [],
                'related_pages': []
            }
            
            # Extract categories
            for cat_link in soup.select('div.catlinks a'):
                if cat_link.text:
                    metadata['categories'].append(cat_link.text.strip())
            
            # Extract "See also" links
            see_also = soup.find('span', id='See_also')
            if see_also:
                section = see_also.find_parent().find_next_sibling()
                if section:
                    for link in section.find_all('a'):
                        if link.text:
                            metadata['related_pages'].append(link.text.strip())
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}")
            return {'source': 'Wikipedia'}
    
    def _extract_relationships(self, soup: BeautifulSoup, theorem_name: str) -> List[Dict[str, Any]]:
        """Extract relationships from the theorem page."""
        relationships = []
        
        try:
            # Look for relationships in the text
            content = soup.select_one('div.mw-parser-output')
            if content:
                text = content.text.lower()
                
                # Look for common relationship patterns
                proof_patterns = [
                    r'proves? that',
                    r'proof of',
                    r'demonstrated by',
                    r'shown by'
                ]
                
                dependency_patterns = [
                    r'depends on',
                    r'requires',
                    r'based on',
                    r'follows from'
                ]
                
                # Find theorem names using common patterns
                theorem_pattern = r'([A-Z][a-zA-Z\s\']+(?:theorem|lemma|conjecture))'
                theorems = re.finditer(theorem_pattern, content.text)
                
                for match in theorems:
                    related_theorem = match.group(1).strip()
                    if related_theorem == theorem_name:
                        continue
                    
                    # Check the context for relationship type
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(content.text), match.end() + 50)
                    context = content.text[context_start:context_end].lower()
                    
                    if any(re.search(pattern, context) for pattern in proof_patterns):
                        relationships.append({
                            'source_theorem': related_theorem,
                            'target_theorem': theorem_name,
                            'relationship_type': 'proves',
                            'confidence': 0.8
                        })
                    elif any(re.search(pattern, context) for pattern in dependency_patterns):
                        relationships.append({
                            'source_theorem': theorem_name,
                            'target_theorem': related_theorem,
                            'relationship_type': 'depends_on',
                            'confidence': 0.7
                        })
                    else:
                        relationships.append({
                            'source_theorem': theorem_name,
                            'target_theorem': related_theorem,
                            'relationship_type': 'related_to',
                            'confidence': 0.5
                        })
            
        except Exception as e:
            self.logger.warning(f"Error extracting relationships: {str(e)}")
        
        return relationships

    def _get_wikipedia_redirect(self, theorem_name: str) -> Optional[str]:
        """Check for Wikipedia redirects for common theorem name variations."""
        # Common redirects for mathematical theorems
        redirect_map = {
            'Fundamental_Theorem_of_Linear_Algebra': 'Rank–nullity_theorem',
            'Fundamental_Theorem_of_Galois_Theory': 'Fundamental_theorem_of_Galois_theory',
            'Fundamental_Theorem_of_Riemannian_Geometry': None,  # Doesn't exist
            'Classification_of_Finite_Simple_Groups': 'Classification_of_finite_simple_groups',
            'Classification_of_Surface_Groups': None,  # Doesn't exist
            'Godel_Incompleteness_Theorem': 'Gödel\'s_incompleteness_theorems',
            'Lowenheim_Skolem_Theorem': 'Löwenheim–Skolem_theorem',
            'Craig_Interpolation_Theorem': 'Craig_interpolation',
            'Lindstrom_Theorem': 'Lindström\'s_theorem',
            'Morley_Theorem': 'Morley\'s_categoricity_theorem',
            # Additional mappings for common 404 errors
            'Fundamental_theorem_of_Galois_theory': 'Fundamental_theorem_of_Galois_theory',
            'Classification_of_finite_simple_groups': 'Classification_of_finite_simple_groups',
            'Rank–nullity_theorem': 'Rank–nullity_theorem',
            'Gödel\'s_incompleteness_theorems': 'Gödel\'s_incompleteness_theorems',
            'Löwenheim–Skolem_theorem': 'Löwenheim–Skolem_theorem',
            'Compactness_theorem': 'Compactness_theorem',
            'Craig_interpolation': 'Craig_interpolation',
            'Lindström\'s_theorem': 'Lindström\'s_theorem',
            'Morley\'s_categoricity_theorem': 'Morley\'s_categoricity_theorem'
        }

        return redirect_map.get(theorem_name)

    def _search_wikipedia(self, theorem_name: str) -> Optional[str]:
        """Search Wikipedia for the theorem and return the best match URL."""
        try:
            search_query = theorem_name.replace('_', ' ')
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': search_query,
                'srlimit': 5
            }

            response = self._make_request(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'query' in data and 'search' in data['query']:
                    results = data['query']['search']
                    if results:
                        # Return the first result that contains "theorem", "lemma", or "conjecture"
                        for result in results:
                            title = result['title']
                            if any(term in title.lower() for term in ['theorem', 'lemma', 'conjecture']):
                                return f"{self.base_url}/{quote(title.replace(' ', '_'))}"

                        # If no theorem-like result, return the first result
                        return f"{self.base_url}/{quote(results[0]['title'].replace(' ', '_'))}"

            return None

        except Exception as e:
            self.logger.warning(f"Error searching Wikipedia for {theorem_name}: {str(e)}")
            return None