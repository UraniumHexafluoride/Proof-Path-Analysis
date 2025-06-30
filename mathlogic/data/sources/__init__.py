"""
Source implementations for mathematical theorem scraping.
Currently focused on Wikipedia as the primary source.
"""

from .base_source import MathSource
from .wikipedia_source import WikipediaSource
# from .proofwiki_source import ProofWikiSource  # Temporarily disabled
# from .nlab_source import NLabSource  # Temporarily disabled
# from .arxiv_source import ArXivSource  # Temporarily disabled

__all__ = ['MathSource', 'WikipediaSource']

# Register source classes - only Wikipedia enabled for now
SOURCES = {
    'wikipedia': WikipediaSource,
    # 'proofwiki': ProofWikiSource,  # Temporarily disabled
    # 'nlab': NLabSource,  # Temporarily disabled
    # 'arxiv': ArXivSource  # Temporarily disabled
} 