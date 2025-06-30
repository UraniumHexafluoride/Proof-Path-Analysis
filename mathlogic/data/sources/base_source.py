"""
Base class for all theorem sources.
"""

import logging
from typing import Dict, Any, Generator, List, Optional, Iterator
import requests
from dataclasses import dataclass
import time
import random

@dataclass
class ScrapedTheorem:
    """Represents a scraped theorem."""
    name: str
    description: str
    source_url: str
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]]

class MathSource:
    """Base class for all theorem sources."""
    
    def __init__(self):
        """Initialize the source."""
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_theorems(self, items: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Fetch theorems from the source.
        
        Args:
            items: Optional list of specific items to fetch (e.g., URLs or IDs)
            
        Returns:
            Iterator of theorem dictionaries
        """
        raise NotImplementedError("Subclasses must implement fetch_theorems")
    
    def process_stream(self, stream: Iterator[Dict[str, Any]], source_name: str) -> Iterator[Dict[str, Any]]:
        """
        Process a stream of theorems.
        
        Args:
            stream: Iterator of theorem dictionaries
            source_name: Name of the source
            
        Returns:
            Iterator of processed theorem dictionaries
        """
        for theorem in stream:
            try:
                # Add source metadata
                if 'metadata' not in theorem:
                    theorem['metadata'] = {}
                theorem['metadata']['source'] = source_name
                theorem['metadata']['timestamp'] = time.time()
                
                # Validate theorem
                if self._validate_theorem(theorem):
                    yield theorem
                else:
                    self.logger.warning(f"Invalid theorem data: {theorem.get('name', 'unknown')}")
                    
            except Exception as e:
                self.logger.error(f"Error processing theorem: {str(e)}")
                continue
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """
        Make an HTTP request with retries and error handling.

        Args:
            url: URL to request
            method: HTTP method to use
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, **kwargs)

                # Don't raise for 404s - let the caller handle them
                if response.status_code == 404:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"Request failed (attempt {attempt + 1}/{max_retries}): "
                            f"{response.status_code} {response.reason} for url: {url}"
                        )
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return response

                # Raise for other HTTP errors
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                    raise
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                time.sleep(retry_delay * (attempt + 1))
    
    def _random_delay(self, min_delay: float = 0.5, max_delay: float = 2.0):
        """Add a random delay between requests to avoid rate limiting."""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from scraped data.
        
        Args:
            data: Raw scraped data
            
        Returns:
            Dictionary of metadata
        """
        return {
            'source': self.__class__.__name__,
            'timestamp': time.time()
        }
    
    def _extract_relationships(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relationships from scraped data.
        
        Args:
            data: Raw scraped data
            
        Returns:
            List of relationship dictionaries
        """
        return []
    
    def _normalize_theorem_name(self, name: str) -> str:
        """
        Normalize theorem name for consistency.
        
        Args:
            name: Original theorem name
            
        Returns:
            Normalized name
        """
        return name.strip()
    
    def _validate_theorem(self, theorem: Dict[str, Any]) -> bool:
        """
        Validate a theorem before yielding it.
        
        Args:
            theorem: Theorem data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = {'name', 'description', 'source_url'}
        return all(field in theorem and theorem[field] for field in required_fields)