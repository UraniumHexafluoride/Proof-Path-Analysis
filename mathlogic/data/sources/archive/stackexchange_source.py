"""
Stack Exchange theorem scraper.
"""

import logging
from stackapi import StackAPI
from typing import Dict, Any, Generator, List, Optional
from .base_source import MathSource

class StackExchangeSource(MathSource):
    """Scraper for Mathematics Stack Exchange content."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Stack Exchange scraper.
        
        Args:
            api_key: Optional Stack Exchange API key
        """
        super().__init__()
        self.api = StackAPI('math', key=api_key)
        self.api.page_size = 100
        self.api.max_pages = 1
    
    def fetch_theorems(self, items: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch theorems from Mathematics Stack Exchange.
        
        Args:
            items: Optional list of tags to search for (e.g., ['theorem', 'proof'])
            
        Yields:
            Dictionary containing theorem data
        """
        try:
            # Use provided tags or default to 'theorems'
            tags = items if items else ['theorems']
            
            # Search for questions with the specified tags
            questions = self.api.fetch('questions', tagged=tags, sort='votes')
            
            for question in questions['items']:
                try:
                    if 'theorem' in question['title'].lower():
                        name = question['title']
                        description = question.get('body', '')
                        metadata = self._extract_metadata(question)
                        
                        yield {
                            'name': name,
                            'description': description,
                            'source_url': f"https://math.stackexchange.com/q/{question['question_id']}",
                            'metadata': metadata
                        }
                        
                    self._random_delay()
                    
                except Exception as e:
                    logging.error(f"Error processing Stack Exchange question {question.get('question_id')}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error fetching from Stack Exchange: {str(e)}")
    
    def _extract_metadata(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a Stack Exchange question.
        
        Args:
            question: Question data from Stack Exchange API
            
        Returns:
            Dictionary of metadata
        """
        return {
            'score': question['score'],
            'tags': question['tags'],
            'answer_count': question['answer_count'],
            'view_count': question['view_count'],
            'source': 'Stack Exchange'
        } 