"""
Test script to verify scraper integration.
"""

from mathlogic.data.multi_source_scraper import MultiSourceScraper
from mathlogic.data.chunk_manager import ChunkManager
from mathlogic.data.stream_processor import StreamProcessor
from mathlogic.core.entailment import EntailmentCone

def main():
    # Initialize components
    scraper = MultiSourceScraper(output_dir="test_output", batch_size=5)
    chunk_manager = ChunkManager("test_output/chunks")
    stream_processor = StreamProcessor("test_output/processed")
    entailment_cone = EntailmentCone("Test Cone")
    
    # Test scraping with just ProofWiki (fastest source)
    print("Testing scraper...")
    theorems = scraper.scrape_all_sources()
    print(f"Found {len(theorems)} theorems")
    
    # Test chunk processing
    print("\nTesting chunk processing...")
    for theorem in theorems.values():
        chunk_manager.process_stream(iter([theorem]), 'test')
    
    # Test relationship building
    print("\nTesting relationship building...")
    relationships_df = stream_processor.build_relationships_stream()
    print(f"Found {len(relationships_df)} relationships")
    
    # Test entailment graph building
    print("\nTesting entailment graph building...")
    graph = stream_processor.build_graph_stream(relationships_df)
    print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 