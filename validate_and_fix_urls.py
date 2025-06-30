"""
URL validation and configuration fixing script.
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class URLValidator:
    """Validates URLs for mathematical theorem sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def validate_wikipedia_urls(self, theorem_names: List[str]) -> Dict[str, Dict]:
        """Validate Wikipedia URLs for theorem names."""
        base_url = "https://en.wikipedia.org/wiki"
        results = {}
        
        for theorem in theorem_names:
            logger.info(f"Validating Wikipedia URL for: {theorem}")
            
            # Try multiple URL formats
            url_formats = [
                theorem,
                theorem.replace('_', ' '),
                theorem.lower(),
                theorem.replace('_', '-'),
                theorem.replace('Theorem', 'theorem'),
                theorem.replace('_of_', '_'),
            ]
            
            # Add known redirects
            redirects = self._get_wikipedia_redirects()
            if theorem in redirects:
                url_formats.insert(0, redirects[theorem])
            
            valid_url = None
            status_codes = []
            
            for url_format in url_formats:
                if url_format is None:
                    continue
                    
                url = f"{base_url}/{quote(url_format)}"
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
                'found': valid_url is not None,
                'valid_url': valid_url,
                'attempts': status_codes
            }
        
        return results
    
    def validate_nlab_urls(self, theorem_names: List[str]) -> Dict[str, Dict]:
        """Validate nLab URLs for theorem names."""
        base_url = "https://ncatlab.org/nlab/show"
        results = {}
        
        for theorem in theorem_names:
            logger.info(f"Validating nLab URL for: {theorem}")
            
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
                'found': valid_url is not None,
                'valid_url': valid_url,
                'attempts': status_codes
            }
        
        return results
    
    def _get_wikipedia_redirects(self) -> Dict[str, str]:
        """Get known Wikipedia redirects."""
        return {
            'Fundamental_Theorem_of_Linear_Algebra': 'Rank–nullity_theorem',
            'Fundamental_Theorem_of_Galois_Theory': 'Fundamental_theorem_of_Galois_theory',
            'Fundamental_Theorem_of_Riemannian_Geometry': None,  # Doesn't exist
            'Classification_of_Finite_Simple_Groups': 'Classification_of_finite_simple_groups',
            'Classification_of_Surface_Groups': None,  # Doesn't exist
            'Godel_Incompleteness_Theorem': 'Gödel\'s_incompleteness_theorems',
            'Lowenheim_Skolem_Theorem': 'Löwenheim–Skolem_theorem',
            'Craig_Interpolation_Theorem': 'Craig_interpolation',
            'Lindstrom_Theorem': 'Lindström\'s_theorem',
            'Morley_Theorem': 'Morley\'s_categoricity_theorem'
        }

def load_config(config_file: str = 'scraping_config.json') -> Dict:
    """Load scraping configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def save_validation_results(results: Dict, output_file: str = 'url_validation_results.json'):
    """Save validation results."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Validation results saved to: {output_file}")

def create_fixed_config(config: Dict, validation_results: Dict) -> Dict:
    """Create a fixed configuration based on validation results."""
    fixed_config = config.copy()
    
    # Fix Wikipedia items
    if 'wikipedia' in config and 'items' in config['wikipedia']:
        wikipedia_results = validation_results.get('wikipedia', {})
        fixed_items = []
        
        for item in config['wikipedia']['items']:
            if item in wikipedia_results:
                if wikipedia_results[item]['found']:
                    fixed_items.append(item)
                else:
                    logger.warning(f"Removing Wikipedia item (not found): {item}")
            else:
                fixed_items.append(item)  # Keep if not validated
        
        fixed_config['wikipedia']['items'] = fixed_items
    
    # Fix nLab items
    if 'nlab' in config and 'items' in config['nlab']:
        nlab_results = validation_results.get('nlab', {})
        fixed_items = []
        
        for item in config['nlab']['items']:
            if item in nlab_results:
                if nlab_results[item]['found']:
                    fixed_items.append(item)
                else:
                    logger.warning(f"Removing nLab item (not found): {item}")
            else:
                fixed_items.append(item)  # Keep if not validated
        
        fixed_config['nlab']['items'] = fixed_items
    
    return fixed_config

def print_validation_summary(results: Dict):
    """Print a summary of validation results."""
    print("\n" + "="*60)
    print("URL VALIDATION SUMMARY")
    print("="*60)
    
    for source, source_results in results.items():
        total = len(source_results)
        found = sum(1 for r in source_results.values() if r['found'])
        not_found = total - found
        
        print(f"\n{source.upper()}:")
        print(f"  Total items: {total}")
        print(f"  Found: {found} ({found/total*100:.1f}%)")
        print(f"  Not found: {not_found} ({not_found/total*100:.1f}%)")
        
        if not_found > 0:
            print(f"  Items not found:")
            for item, result in source_results.items():
                if not result['found']:
                    print(f"    - {item}")

def main():
    """Main validation and fixing process."""
    print("🔍 URL Validator and Configuration Fixer")
    print("="*50)
    
    # Load configuration
    config = load_config()
    validator = URLValidator()
    
    validation_results = {}
    
    # Validate Wikipedia URLs
    if config.get('wikipedia', {}).get('enabled', False):
        wikipedia_items = config['wikipedia'].get('items', [])
        if wikipedia_items:
            print(f"\n📖 Validating {len(wikipedia_items)} Wikipedia URLs...")
            validation_results['wikipedia'] = validator.validate_wikipedia_urls(wikipedia_items)
    
    # Validate nLab URLs
    if config.get('nlab', {}).get('enabled', False):
        nlab_items = config['nlab'].get('items', [])
        if nlab_items:
            print(f"\n🔬 Validating {len(nlab_items)} nLab URLs...")
            validation_results['nlab'] = validator.validate_nlab_urls(nlab_items)
    
    # Save validation results
    save_validation_results(validation_results)
    
    # Print summary
    print_validation_summary(validation_results)
    
    # Create fixed configuration
    fixed_config = create_fixed_config(config, validation_results)
    
    # Save fixed configuration
    with open('scraping_config_fixed.json', 'w') as f:
        json.dump(fixed_config, f, indent=4)
    
    print(f"\n✅ Fixed configuration saved to: scraping_config_fixed.json")
    print("🚀 You can now run the scraper with the fixed configuration!")

if __name__ == '__main__':
    main()
