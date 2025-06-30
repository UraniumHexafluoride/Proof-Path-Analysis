#!/usr/bin/env python3
"""
Script to validate URLs before running the full scraping process.
This helps identify problematic URLs and suggests fixes.
"""

import sys
import logging
from mathlogic.utils.url_validator import validate_scraping_config

def main():
    """Main entry point for URL validation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("URL Validation Tool")
    print("=" * 50)
    print("This tool validates URLs in your scraping configuration")
    print("to identify problematic links before running the full scraper.\n")
    
    # Get config file from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'scraping_config.json'
    
    try:
        validate_scraping_config(config_file)
        print("\nValidation complete!")
        print("Check the detailed results in: entailment_output/scraped_data/url_validation_results.json")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
