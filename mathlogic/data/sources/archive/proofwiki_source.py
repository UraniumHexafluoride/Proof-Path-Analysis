"""
ProofWiki theorem scraper.
"""

import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, Any, Iterator, List, Optional
from urllib.parse import quote, urljoin
from .base_source import MathSource

class ProofWikiSource(MathSource):
    """Scraper for ProofWiki."""
    
    def __init__(self):
        """Initialize the ProofWiki scraper."""
        super().__init__()
        self.base_url = "https://proofwiki.org"
        self.logger = logging.getLogger(__name__)
    
    def fetch_theorems(self, items: Optional[List[str]] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Fetch theorems from ProofWiki.
        
        Args:
            items: Optional list of specific theorem names to fetch
            config: Optional configuration dictionary with additional settings
            
        Returns:
            Iterator of theorem dictionaries
        """
        try:
            # Get configuration
            batch_size = config.get('batch_size', 10) if config else 10
            parallel_requests = config.get('parallel_requests', 3) if config else 3
            rate_limit = config.get('rate_limit', 1.5) if config else 1.5
            
            # Combine items from both sources
            all_items = []
            if items:
                all_items.extend(items)
            if config and 'items' in config:
                all_items.extend(config['items'])
            
            # Process categories if present
            if config and 'categories' in config:
                for category in config['categories']:
                    try:
                        # Format category URL
                        category_name = category.replace(' ', '_')
                        category_url = f"{self.base_url}/wiki/Category:{quote(category_name, safe='')}"
                        
                        response = self._make_request(category_url)
                        if response.status_code == 404:
                            # Try alternate URL format
                            category_url = f"{self.base_url}/wiki/Category:{quote(category)}"
                            response = self._make_request(category_url)
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find theorem links in category
                        for link in soup.select('div.mw-category a, div#mw-pages a'):
                            if link.get('href') and any(term in link.text.lower() for term in ['theorem', 'lemma', 'conjecture']):
                                all_items.append(link.text.strip())
                                
                    except Exception as e:
                        self.logger.error(f"Error processing category {category}: {str(e)}")
                        continue
            
            # Remove duplicates while preserving order
            all_items = list(dict.fromkeys(all_items))
            
            # Create event loop and process items
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def process_all():
                async with aiohttp.ClientSession() as session:
                    results = []
                    for i in range(0, len(all_items), batch_size):
                        batch = all_items[i:i + batch_size]
                        batch_results = await self._process_batch(session, batch, rate_limit)
                        results.extend(batch_results)
                        await asyncio.sleep(rate_limit)  # Rate limiting between batches
                    return results
            
            results = loop.run_until_complete(process_all())
            loop.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fetch_theorems: {e}")
            return []
    
    async def _process_batch(self, session: aiohttp.ClientSession, items: List[str], rate_limit: float) -> List[Dict[str, Any]]:
        """Process a batch of items concurrently."""
        tasks = []
        for item in items:
            task = asyncio.ensure_future(self._process_item(session, item, rate_limit))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]
    
    async def _process_item(self, session: aiohttp.ClientSession, item: str, rate_limit: float) -> Optional[Dict[str, Any]]:
        """Process a single theorem item."""
        try:
            # Format theorem URL
            theorem_name = item.replace(' ', '_')
            theorem_url = f"{self.base_url}/wiki/{quote(theorem_name, safe='')}"
            
            async with session.get(theorem_url) as response:
                if response.status == 404:
                    # Try alternate URL format
                    theorem_url = f"{self.base_url}/wiki/{quote(item)}"
                    async with session.get(theorem_url) as response2:
                        if response2.status == 404:
                            self.logger.warning(f"Theorem not found: {item}")
                            return None
                        text = await response2.text()
                else:
                    text = await response.text()
                
                soup = BeautifulSoup(text, 'html.parser')
                
                # Extract theorem data
                description = self._extract_description(soup)
                if not description:
                    return None
                
                metadata = self._extract_metadata(soup)
                relationships = self._extract_relationships(soup, item)
                
                await asyncio.sleep(rate_limit)  # Rate limiting
                
                return {
                    'name': item,
                    'description': description,
                    'source_url': theorem_url,
                    'metadata': metadata,
                    'relationships': relationships
                }
                
        except Exception as e:
            self.logger.error(f"Error processing theorem {item}: {str(e)}")
            return None
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract theorem description from the page."""
        try:
            # Find the theorem statement section
            content = soup.select_one('div#mw-content-text')
            if content:
                # Look for theorem statement markers
                for h in content.find_all(['h2', 'h3', 'h4']):
                    if any(word in h.text.lower() for word in ['statement', 'theorem', 'definition']):
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