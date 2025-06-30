"""
Main script for running the chunked theorem scraping system.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from datetime import datetime
import codecs

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

from mathlogic.data.sources.proofwiki_source import ProofWikiSource
from mathlogic.data.sources.wikipedia_source import WikipediaSource
from mathlogic.data.sources.nlab_source import NLabSource

# Add file handler for logging
os.makedirs(os.path.join('entailment_output', 'scraped_data'), exist_ok=True)
log_file = os.path.join('entailment_output', 'scraped_data', f'scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load scraping configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def get_source_instance(source_name: str) -> Any:
    """Get source scraper instance based on name."""
    sources = {
        'proofwiki': ProofWikiSource,
        'wikipedia': WikipediaSource,
        'nlab': NLabSource
    }
    
    if source_name not in sources:
        raise ValueError(f"Unknown source: {source_name}")
    
    return sources[source_name]()

def process_source(source_name: str, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Process a single source."""
    try:
        source_config = config.get(source_name, {})
        if not source_config.get('enabled', False):
            return
        
        items = source_config.get('items', [])
        categories = source_config.get('categories', [])
        
        if not items and not categories:
            logger.warning(f"No items or categories configured for {source_name}")
            return
        
        source = get_source_instance(source_name)
        
        print(f"\nProcessing source: {source_name}")
        print(f"Items to process: {len(items)}")
        print(f"Categories to process: {len(categories)}")
        
        successful = 0
        failed = 0
        total_theorems = 0
        
        # Create progress bars
        with tqdm(total=len(items) + len(categories), desc=f"Scraping {source_name}", unit="item") as pbar:
            # Process specific items
            if items:
                for theorem in source.fetch_theorems(items=items, config=source_config):
                    if theorem:
                        results[source_name].append(theorem)
                        successful += 1
                        total_theorems += 1
                        if 'name' in theorem:
                            logger.info(f"[SUCCESS] Successfully scraped: {theorem['name']}")
                    else:
                        failed += 1
                    pbar.update(1)
            
            # Process categories
            if categories:
                for theorem in source.fetch_theorems(config=source_config):
                    if theorem:
                        results[source_name].append(theorem)
                        successful += 1
                        total_theorems += 1
                        if 'name' in theorem:
                            logger.info(f"[SUCCESS] Successfully scraped: {theorem['name']}")
                    else:
                        failed += 1
                    # Update progress bar for each theorem found in categories
                    pbar.total = pbar.n + 1
                    pbar.update(1)
        
        print(f"\n{source_name} Summary:")
        print(f"- Total items processed: {len(items) + len(categories)}")
        print(f"- Total theorems found: {total_theorems}")
        print(f"- Successful: {successful}")
        print(f"- Failed: {failed}")
        print()
        
    except Exception as e:
        logger.error(f"Error processing source {source_name}: {str(e)}")
        raise

def save_results(results: Dict[str, List[Dict[str, Any]]], config: Dict[str, Any]) -> None:
    """Save scraping results."""
    try:
        output_dir = config.get('output_dir', 'scraped_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'results_{timestamp}.json')
        
        # Count total theorems and relationships
        total_theorems = sum(len(theorems) for theorems in results.values())
        total_relationships = sum(
            sum(len(theorem.get('relationships', [])) for theorem in theorems)
            for theorems in results.values()
        )
        
        # Add summary to results
        summary = {
            'timestamp': timestamp,
            'total_theorems': total_theorems,
            'total_relationships': total_relationships,
            'source_counts': {
                source: len(theorems)
                for source, theorems in results.items()
            }
        }
        
        final_results = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Total theorems collected: {total_theorems}")
        logger.info(f"Total relationships found: {total_relationships}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main() -> None:
    """Main entry point for the scraping system."""
    try:
        print("Loading configuration...")
        
        # Load configuration
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'scraping_config.json'
        config = load_config(config_file)
        
        # Initialize results dictionary
        results = {
            'proofwiki': [],
            'wikipedia': [],
            'nlab': []
        }
        
        # Get enabled sources
        enabled_sources = [
            source_name for source_name, source_config in config.items()
            if isinstance(source_config, dict) and source_config.get('enabled', False)
        ]
        
        print(f"\nStarting scraping process with {len(enabled_sources)} enabled sources:")
        print("\n" + "="*50)
        
        start_time = time.time()
        
        # Process each source
        for source_name in enabled_sources:
            if source_name in results:  # Only process known sources
                print("-" * 50)
                process_source(source_name, config, results)
        
        # Save final results
        save_results(results, config)
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        total_successful = sum(len(theorems) for theorems in results.values())
        total_relationships = sum(
            sum(len(theorem.get('relationships', [])) for theorem in theorems)
            for theorems in results.values()
        )
        
        print("=" * 50)
        print("Scraping Complete!")
        print(f"Total time: {duration/60:.2f} minutes")
        print(f"Total theorems collected: {total_successful}")
        print(f"Total relationships found: {total_relationships}")
        print("=" * 50 + "\n")
        
        logger.info("Scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")
        raise

if __name__ == '__main__':
    main() 