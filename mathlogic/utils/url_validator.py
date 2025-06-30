"""
URL validation utility for theorem scraping.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import time

class URLValidator:
    """Validates URLs before scraping to identify problematic links."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mathematical Research Bot (Academic Use)'
        })
    
    def validate_wikipedia_urls(self, theorem_names: List[str]) -> Dict[str, Dict]:
        """
        Validate Wikipedia URLs for a list of theorem names.
        
        Args:
            theorem_names: List of theorem names to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        base_url = "https://en.wikipedia.org/wiki"
        
        for theorem in theorem_names:
            self.logger.info(f"Validating Wikipedia URL for: {theorem}")
            
            # Try multiple URL formats
            url_formats = [
                theorem.replace(' ', '_'),
                theorem.replace(' ', '-'),
                theorem.lower().replace(' ', '_'),
                theorem.replace('_', ' '),
            ]
            
            valid_url = None
            status_codes = []
            
            for url_format in url_formats:
                url = f"{base_url}/{quote(url_format)}"
                try:
                    response = self.session.head(url, timeout=10)
                    status_codes.append((url_format, response.status_code))
                    
                    if response.status_code == 200:
                        valid_url = url
                        break
                    elif response.status_code == 301 or response.status_code == 302:
                        # Follow redirect
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            valid_url = response.url
                            break
                            
                except Exception as e:
                    status_codes.append((url_format, f"Error: {str(e)}"))
                
                time.sleep(0.5)  # Rate limiting
            
            results[theorem] = {
                'valid_url': valid_url,
                'status_codes': status_codes,
                'found': valid_url is not None
            }
            
            if valid_url:
                self.logger.info(f"✓ Found valid URL: {valid_url}")
            else:
                self.logger.warning(f"✗ No valid URL found for: {theorem}")
        
        return results
    
    def validate_nlab_urls(self, theorem_names: List[str]) -> Dict[str, Dict]:
        """
        Validate nLab URLs for a list of theorem names.
        
        Args:
            theorem_names: List of theorem names to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        base_url = "https://ncatlab.org/nlab/show"
        
        for theorem in theorem_names:
            self.logger.info(f"Validating nLab URL for: {theorem}")
            
            # Try multiple URL formats for nLab
            url_formats = [
                theorem.lower().replace(' ', '+'),
                theorem.replace(' ', '+'),
                theorem.lower().replace(' ', '_'),
                theorem.replace(' ', '_'),
                theorem.replace(' ', '+').replace('_', '+'),
            ]
            
            valid_url = None
            status_codes = []
            
            for url_format in url_formats:
                url = f"{base_url}/{url_format}"
                try:
                    response = self.session.head(url, timeout=10)
                    status_codes.append((url_format, response.status_code))
                    
                    if response.status_code == 200:
                        valid_url = url
                        break
                        
                except Exception as e:
                    status_codes.append((url_format, f"Error: {str(e)}"))
                
                time.sleep(0.5)  # Rate limiting
            
            results[theorem] = {
                'valid_url': valid_url,
                'status_codes': status_codes,
                'found': valid_url is not None
            }
            
            if valid_url:
                self.logger.info(f"✓ Found valid URL: {valid_url}")
            else:
                self.logger.warning(f"✗ No valid URL found for: {theorem}")
        
        return results
    
    def generate_validation_report(self, wikipedia_results: Dict, nlab_results: Dict) -> str:
        """Generate a validation report."""
        report = []
        report.append("URL Validation Report")
        report.append("=" * 50)
        
        # Wikipedia results
        report.append("\nWikipedia Results:")
        report.append("-" * 20)
        wiki_found = sum(1 for r in wikipedia_results.values() if r['found'])
        wiki_total = len(wikipedia_results)
        report.append(f"Found: {wiki_found}/{wiki_total} ({wiki_found/wiki_total*100:.1f}%)")
        
        report.append("\nMissing Wikipedia pages:")
        for theorem, result in wikipedia_results.items():
            if not result['found']:
                report.append(f"  - {theorem}")
        
        # nLab results
        report.append("\nnLab Results:")
        report.append("-" * 20)
        nlab_found = sum(1 for r in nlab_results.values() if r['found'])
        nlab_total = len(nlab_results)
        report.append(f"Found: {nlab_found}/{nlab_total} ({nlab_found/nlab_total*100:.1f}%)")
        
        report.append("\nMissing nLab pages:")
        for theorem, result in nlab_results.items():
            if not result['found']:
                report.append(f"  - {theorem}")
        
        return "\n".join(report)

def validate_scraping_config(config_file: str = "scraping_config.json") -> None:
    """Validate URLs in the scraping configuration."""
    import json
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    validator = URLValidator()
    
    # Validate Wikipedia URLs
    if 'wikipedia' in config and config['wikipedia'].get('enabled', False):
        wikipedia_items = config['wikipedia'].get('items', [])
        if wikipedia_items:
            print("Validating Wikipedia URLs...")
            wikipedia_results = validator.validate_wikipedia_urls(wikipedia_items)
        else:
            wikipedia_results = {}
    else:
        wikipedia_results = {}
    
    # Validate nLab URLs
    if 'nlab' in config and config['nlab'].get('enabled', False):
        nlab_items = config['nlab'].get('items', [])
        if nlab_items:
            print("Validating nLab URLs...")
            nlab_results = validator.validate_nlab_urls(nlab_items)
        else:
            nlab_results = {}
    else:
        nlab_results = {}
    
    # Generate and print report
    report = validator.generate_validation_report(wikipedia_results, nlab_results)
    print("\n" + report)
    
    # Save detailed results
    import os
    os.makedirs('entailment_output/scraped_data', exist_ok=True)
    
    with open('entailment_output/scraped_data/url_validation_results.json', 'w') as f:
        json.dump({
            'wikipedia': wikipedia_results,
            'nlab': nlab_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: entailment_output/scraped_data/url_validation_results.json")

if __name__ == "__main__":
    validate_scraping_config()
