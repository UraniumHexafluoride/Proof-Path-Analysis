"""
Multi-source theorem scraper with efficient batch processing and rate limiting.
"""

import logging
import time
from typing import Dict, List, Any
from pathlib import Path

class MultiSourceScraper:
    """Scrapes theorems from multiple sources with batching and rate limiting."""
    
    def __init__(self, output_dir: str = None, batch_size: int = 10, rate_limit: float = 1.0):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Optional directory for saving scraped data
            batch_size: Number of theorems to process in each batch
            rate_limit: Time to wait between batches in seconds
        """
        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self._setup_sources()
    
    def _setup_sources(self):
        """Set up the theorem sources."""
        self.sources = {
            'proofwiki': self._create_proofwiki_source(),
            'arxiv': self._create_arxiv_source(),
            'stackexchange': self._create_stackexchange_source(),
            'nlab': self._create_nlab_source(),
            'wikipedia': self._create_wikipedia_source()
        }
    
    def scrape_all_sources(self) -> Dict[str, Any]:
        """
        Scrape theorems from all sources with batching and rate limiting.
        
        Returns:
            Dict mapping theorem names to their data
        """
        all_theorems = {}
        
        for source_name, source in self.sources.items():
            logging.info(f"Scraping from {source_name}...")
            try:
                theorems_batch = []
                for theorem in source.fetch_theorems():
                    theorems_batch.append(theorem)
                    if len(theorems_batch) >= self.batch_size:
                        self._process_batch(theorems_batch, all_theorems)
                        theorems_batch = []
                        time.sleep(self.rate_limit)
                
                # Process remaining theorems
                if theorems_batch:
                    self._process_batch(theorems_batch, all_theorems)
                
                logging.info(f"Completed {source_name}: {len(all_theorems)} theorems found")
            except Exception as e:
                logging.error(f"Error scraping from {source_name}: {str(e)}")
                continue
        
        if self.output_dir:
            self._save_results(all_theorems)
        
        return all_theorems
    
    def _process_batch(self, theorems_batch: List[Dict], all_theorems: Dict):
        """
        Process a batch of theorems.
        
        Args:
            theorems_batch: List of theorem dictionaries
            all_theorems: Dictionary to store processed theorems
        """
        for theorem in theorems_batch:
            try:
                normalized_name = self._normalize_theorem_name(theorem['name'])
                if normalized_name not in all_theorems:
                    all_theorems[normalized_name] = theorem
                    logging.info(f"Processed theorem: {theorem['name']} -> {normalized_name}")
            except Exception as e:
                logging.error(f"Error processing theorem {theorem.get('name', 'unknown')}: {str(e)}")
    
    def _normalize_theorem_name(self, name: str) -> str:
        """
        Normalize theorem name for consistency.
        
        Args:
            name: Original theorem name
            
        Returns:
            Normalized name
        """
        return name.lower().strip()
    
    def _save_results(self, theorems: Dict):
        """
        Save scraped theorems to output directory.
        
        Args:
            theorems: Dictionary of theorem data
        """
        if not self.output_dir:
            return
            
        output_file = Path(self.output_dir) / f"theorems_{int(time.time())}.json"
        try:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(theorems, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved results to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
    
    def _create_proofwiki_source(self):
        """Create ProofWiki source."""
        from .sources.proofwiki_source import ProofWikiSource
        return ProofWikiSource()
    
    def _create_arxiv_source(self):
        """Create arXiv source."""
        from .sources.arxiv_source import ArxivSource
        return ArxivSource()
    
    def _create_stackexchange_source(self):
        """Create Stack Exchange source."""
        from .sources.stackexchange_source import StackExchangeSource
        return StackExchangeSource()
    
    def _create_nlab_source(self):
        """Create nLab source."""
        from .sources.nlab_source import NLabSource
        return NLabSource()
    
    def _create_wikipedia_source(self):
        """Create Wikipedia source."""
        from .sources.wikipedia_source import WikipediaSource
        return WikipediaSource()

def main():
    scraper = MultiSourceScraper()
    scraper.scrape_all_sources()

if __name__ == "__main__":
    main() 