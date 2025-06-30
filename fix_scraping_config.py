#!/usr/bin/env python3
"""
Script to automatically fix scraping configuration based on URL validation results.
"""

import json
import sys
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_validation_results(results_file: str = 'entailment_output/scraped_data/url_validation_results.json') -> Dict:
    """Load URL validation results."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Validation results file not found: {results_file}")
        logger.info("Please run 'python validate_urls.py' first")
        sys.exit(1)

def get_wikipedia_fixes() -> Dict[str, str]:
    """Get known fixes for Wikipedia URLs."""
    return {
        'Fundamental_Theorem_of_Linear_Algebra': 'Rank–nullity_theorem',
        'Fundamental_Theorem_of_Galois_Theory': 'Fundamental_theorem_of_Galois_theory',
        'Classification_of_Finite_Simple_Groups': 'Classification_of_finite_simple_groups',
        'Godel_Incompleteness_Theorem': 'Gödel\'s_incompleteness_theorems',
        'Lowenheim_Skolem_Theorem': 'Löwenheim–Skolem_theorem',
        'Craig_Interpolation_Theorem': 'Craig_interpolation',
        'Lindstrom_Theorem': 'Lindström\'s_theorem',
        'Morley_Theorem': 'Morley\'s_categoricity_theorem'
    }

def get_nlab_fixes() -> Dict[str, str]:
    """Get known fixes for nLab URLs."""
    return {
        'fundamental theorem of linear algebra': 'rank-nullity theorem',
        'fundamental theorem of riemannian geometry': None,  # Remove - doesn't exist
        'classification of finite simple groups': None,  # Remove - doesn't exist
        'classification of surface groups': None,  # Remove - doesn't exist
    }

def fix_scraping_config(config_file: str = 'scraping_config.json', 
                       validation_file: str = 'entailment_output/scraped_data/url_validation_results.json') -> None:
    """Fix the scraping configuration based on validation results."""
    
    # Load current config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load validation results
    validation_results = load_validation_results(validation_file)
    
    # Get fix mappings
    wikipedia_fixes = get_wikipedia_fixes()
    nlab_fixes = get_nlab_fixes()
    
    changes_made = []
    
    # Fix Wikipedia items
    if 'wikipedia' in config and 'items' in config['wikipedia']:
        original_items = config['wikipedia']['items'][:]
        new_items = []
        
        for item in original_items:
            if item in wikipedia_fixes:
                if wikipedia_fixes[item] is not None:
                    new_items.append(wikipedia_fixes[item])
                    changes_made.append(f"Wikipedia: {item} → {wikipedia_fixes[item]}")
                else:
                    changes_made.append(f"Wikipedia: Removed {item} (doesn't exist)")
            else:
                # Check validation results
                wiki_results = validation_results.get('wikipedia', {})
                if item in wiki_results and wiki_results[item]['found']:
                    new_items.append(item)
                elif item in wiki_results:
                    # Try to find a valid URL from the validation results
                    valid_url = wiki_results[item].get('valid_url')
                    if valid_url:
                        # Extract the corrected name from the URL
                        corrected_name = valid_url.split('/')[-1]
                        new_items.append(corrected_name)
                        changes_made.append(f"Wikipedia: {item} → {corrected_name}")
                    else:
                        changes_made.append(f"Wikipedia: Removed {item} (not found)")
                else:
                    new_items.append(item)  # Keep if not validated
        
        config['wikipedia']['items'] = new_items
    
    # Fix nLab items
    if 'nlab' in config and 'items' in config['nlab']:
        original_items = config['nlab']['items'][:]
        new_items = []
        
        for item in original_items:
            if item in nlab_fixes:
                if nlab_fixes[item] is not None:
                    new_items.append(nlab_fixes[item])
                    changes_made.append(f"nLab: {item} → {nlab_fixes[item]}")
                else:
                    changes_made.append(f"nLab: Removed {item} (doesn't exist)")
            else:
                # Check validation results
                nlab_results = validation_results.get('nlab', {})
                if item in nlab_results and nlab_results[item]['found']:
                    new_items.append(item)
                elif item in nlab_results:
                    changes_made.append(f"nLab: Removed {item} (not found)")
                else:
                    new_items.append(item)  # Keep if not validated
        
        config['nlab']['items'] = new_items
    
    # Save the fixed configuration
    backup_file = config_file.replace('.json', '_backup.json')
    with open(backup_file, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Backup saved to: {backup_file}")
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Report changes
    print("Configuration Fix Report")
    print("=" * 50)
    
    if changes_made:
        print(f"Made {len(changes_made)} changes:")
        for change in changes_made:
            print(f"  - {change}")
    else:
        print("No changes needed - configuration is already optimal!")
    
    print(f"\nUpdated configuration saved to: {config_file}")
    print(f"Backup of original saved to: {backup_file}")

def main():
    """Main entry point."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'scraping_config.json'
    
    try:
        fix_scraping_config(config_file)
        print("\nConfiguration fix complete!")
        print("You can now run the scraper with improved URLs.")
        
    except Exception as e:
        logger.error(f"Error fixing configuration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
