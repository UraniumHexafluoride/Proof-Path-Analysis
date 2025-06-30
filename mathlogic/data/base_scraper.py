"""
Base classes for mathematical content scraping.
"""

import requests
import time
import random
from typing import Dict, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ScrapedTheorem:
    name: str
    description: str
    source_url: str
    metadata: Dict[str, Any]

class MathSource(ABC):
    """Abstract base class for mathematical content sources."""
    
    def __init__(self, delay_range: tuple = (3, 7)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mathematical Research Bot (Academic Use) - Contact: your@email.com'
        })
    
    @abstractmethod
    def fetch_theorems(self) -> Generator[ScrapedTheorem, None, None]:
        """Fetch theorems from the source."""
        pass
    
    def _random_delay(self):
        """Implement random delay between requests."""
        time.sleep(random.uniform(*self.delay_range)) 