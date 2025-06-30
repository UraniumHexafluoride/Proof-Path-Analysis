"""
Script to fix the specific scraping errors encountered.
"""

import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_problematic_items() -> Dict[str, List[str]]:
    """Get the list of items that are causing 404 errors."""
    return {
        'wikipedia': [
            'Fundamental_Theorem_of_Linear_Algebra',  # Should be 'Rank–nullity_theorem'
            'Fundamental_Theorem_of_Galois_Theory',   # Should be 'Fundamental_theorem_of_Galois_theory'
            'Fundamental_Theorem_of_Riemannian_Geometry',  # Doesn't exist
            'Classification_of_Finite_Simple_Groups',  # Should be 'Classification_of_finite_simple_groups'
            'Classification_of_Surface_Groups'  # Doesn't exist
        ],
        'nlab': [
            'set theory',  # Category doesn't exist
            'category theory',  # Category doesn't exist
            'homotopy theory',  # Category doesn't exist
            'fundamental theorem of arithmetic',  # Doesn't exist on nLab
            'fundamental theorem of algebra',  # Doesn't exist on nLab
            'fundamental theorem of calculus',  # Doesn't exist on nLab
            'fundamental theorem of galois theory',  # Doesn't exist on nLab
            'fundamental theorem of linear algebra',  # Doesn't exist on nLab
            'fundamental theorem of homological algebra',  # Doesn't exist on nLab
            'fundamental theorem of morse theory'  # Doesn't exist on nLab
        ]
    }

def get_replacements() -> Dict[str, Dict[str, str]]:
    """Get replacement mappings for problematic items."""
    return {
        'wikipedia': {
            'Fundamental_Theorem_of_Linear_Algebra': 'Rank–nullity_theorem',
            'Fundamental_Theorem_of_Galois_Theory': 'Fundamental_theorem_of_Galois_theory',
            'Classification_of_Finite_Simple_Groups': 'Classification_of_finite_simple_groups',
            # Items to remove (set to None)
            'Fundamental_Theorem_of_Riemannian_Geometry': None,
            'Classification_of_Surface_Groups': None
        },
        'nlab': {
            # Remove problematic categories
            'set theory': None,
            'category theory': None,
            'homotopy theory': None,
            # Remove items that don't exist on nLab
            'fundamental theorem of arithmetic': None,
            'fundamental theorem of algebra': None,
            'fundamental theorem of calculus': None,
            'fundamental theorem of galois theory': None,
            'fundamental theorem of linear algebra': None,
            'fundamental theorem of homological algebra': None,
            'fundamental theorem of morse theory': None
        }
    }

def fix_configuration(config_file: str = 'scraping_config.json') -> Dict:
    """Fix the scraping configuration by removing/replacing problematic items."""
    
    # Load current configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    replacements = get_replacements()
    changes_made = []
    
    # Fix Wikipedia items
    if 'wikipedia' in config and 'items' in config['wikipedia']:
        original_items = config['wikipedia']['items'][:]
        new_items = []
        
        for item in original_items:
            if item in replacements['wikipedia']:
                replacement = replacements['wikipedia'][item]
                if replacement is not None:
                    new_items.append(replacement)
                    changes_made.append(f"Wikipedia: {item} → {replacement}")
                else:
                    changes_made.append(f"Wikipedia: Removed {item} (doesn't exist)")
            else:
                new_items.append(item)
        
        config['wikipedia']['items'] = new_items
    
    # Fix nLab categories
    if 'nlab' in config and 'categories' in config['nlab']:
        original_categories = config['nlab']['categories'][:]
        new_categories = []
        
        for category in original_categories:
            if category in replacements['nlab']:
                replacement = replacements['nlab'][category]
                if replacement is not None:
                    new_categories.append(replacement)
                    changes_made.append(f"nLab categories: {category} → {replacement}")
                else:
                    changes_made.append(f"nLab categories: Removed {category} (doesn't exist)")
            else:
                new_categories.append(category)
        
        config['nlab']['categories'] = new_categories
    
    # Fix nLab items
    if 'nlab' in config and 'items' in config['nlab']:
        original_items = config['nlab']['items'][:]
        new_items = []
        
        for item in original_items:
            if item in replacements['nlab']:
                replacement = replacements['nlab'][item]
                if replacement is not None:
                    new_items.append(replacement)
                    changes_made.append(f"nLab items: {item} → {replacement}")
                else:
                    changes_made.append(f"nLab items: Removed {item} (doesn't exist)")
            else:
                new_items.append(item)
        
        config['nlab']['items'] = new_items
    
    return config, changes_made

def optimize_configuration(config: Dict) -> Dict:
    """Apply additional optimizations to the configuration."""
    
    # Increase rate limits to reduce server load
    if 'wikipedia' in config:
        config['wikipedia']['rate_limit'] = 2.0  # Slower rate
    
    if 'nlab' in config:
        config['nlab']['rate_limit'] = 3.0  # Even slower for nLab
    
    # Add better retry settings
    for source in ['wikipedia', 'nlab', 'proofwiki']:
        if source in config:
            config[source]['max_retries'] = 2
            config[source]['retry_delay'] = 1.0
    
    return config

def create_minimal_test_config(config: Dict) -> Dict:
    """Create a minimal configuration for testing."""
    test_config = {
        'output_dir': 'entailment_output/scraped_data',
        'chunk_size': 10,
        'max_workers': 2,
        'rate_limit': 2.0
    }
    
    # Add only a few reliable items for each source
    test_config['wikipedia'] = {
        'enabled': True,
        'rate_limit': 2.0,
        'max_retries': 2,
        'items': [
            'Fundamental_Theorem_of_Arithmetic',
            'Fundamental_Theorem_of_Algebra',
            'Fundamental_Theorem_of_Calculus',
            'Pythagorean_theorem',
            'Fermat\'s_Last_Theorem'
        ]
    }
    
    test_config['nlab'] = {
        'enabled': True,
        'rate_limit': 3.0,
        'max_retries': 2,
        'categories': [
            'foundations',
            'logic'
        ],
        'items': [
            'yoneda lemma',
            'adjoint functor theorems',
            'stone duality'
        ]
    }
    
    test_config['proofwiki'] = {
        'enabled': True,
        'rate_limit': 1.5,
        'max_retries': 2,
        'items': [
            'Pythagorean_Theorem',
            'Fundamental_Theorem_of_Arithmetic',
            'Binomial_Theorem'
        ]
    }
    
    return test_config

def main():
    """Main fixing process."""
    print("🔧 Fixing Scraping Configuration")
    print("="*50)
    
    # Create backup of original configuration
    import shutil
    shutil.copy('scraping_config.json', 'scraping_config_backup.json')
    logger.info("✅ Backup created: scraping_config_backup.json")
    
    # Fix the configuration
    fixed_config, changes_made = fix_configuration()
    
    # Apply optimizations
    fixed_config = optimize_configuration(fixed_config)
    
    # Save fixed configuration
    with open('scraping_config_fixed.json', 'w') as f:
        json.dump(fixed_config, f, indent=4)
    
    # Create minimal test configuration
    test_config = create_minimal_test_config(fixed_config)
    with open('scraping_config_test.json', 'w') as f:
        json.dump(test_config, f, indent=4)
    
    # Report changes
    print("\n📋 Configuration Fix Report")
    print("-" * 40)
    
    if changes_made:
        print(f"✅ Made {len(changes_made)} changes:")
        for change in changes_made:
            print(f"  • {change}")
    else:
        print("ℹ️  No changes needed - configuration is already optimal!")
    
    print(f"\n📁 Files created:")
    print(f"  • scraping_config_backup.json (backup of original)")
    print(f"  • scraping_config_fixed.json (fixed configuration)")
    print(f"  • scraping_config_test.json (minimal test configuration)")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Test with minimal config: python run_improved_scraping.py scraping_config_test.json")
    print(f"  2. If successful, use full config: python run_improved_scraping.py scraping_config_fixed.json")
    print(f"  3. Monitor logs for any remaining issues")

if __name__ == '__main__':
    main()
