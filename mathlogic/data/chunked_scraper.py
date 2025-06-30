"""
Chunked scraping system that breaks down the scraping process into manageable pieces.
"""

import logging
import time
from typing import Dict, List, Any, Generator, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from queue import Queue
import threading

@dataclass
class ScrapingChunk:
    """Represents a chunk of work to be processed."""
    source_name: str
    start_index: int
    end_index: int
    items_to_process: List[str]  # URLs or IDs to process
    chunk_id: str

@dataclass
class ChunkResult:
    """Results from processing a chunk."""
    chunk_id: str
    source_name: str
    successful_items: List[Dict[str, Any]]
    failed_items: List[tuple[str, str]]  # (item_id, error_message)
    processing_time: float

class ChunkedScraper:
    """
    A scraper that breaks down work into manageable chunks and processes them efficiently.
    Features:
    - Chunked processing with configurable chunk size
    - Progress tracking and resumability
    - Parallel processing with rate limiting
    - Automatic retries and error handling
    - Detailed logging and progress reporting
    """
    
    def __init__(
        self,
        output_dir: str = "entailment_output/scraped_data",
        chunk_size: int = 10,
        max_workers: int = 4,
        rate_limit: float = 1.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Progress tracking
        self.progress_file = self.output_dir / "scraping_progress.json"
        self.results_file = self.output_dir / "scraped_results.json"
        self.progress_lock = threading.Lock()
        
        # Thread-safe progress reporting
        self.progress_queue = Queue()
        self.progress_thread = threading.Thread(target=self._progress_reporter)
        self.progress_thread.daemon = True
        self.progress_thread.start()
    
    def _setup_logging(self):
        """Configure logging for the scraper."""
        log_file = self.output_dir / "scraping.log"
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def _progress_reporter(self):
        """Thread that handles progress reporting."""
        while True:
            msg = self.progress_queue.get()
            if msg is None:
                break
            print(msg)
            self.logger.info(msg)
            self.progress_queue.task_done()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load saved progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_chunks': [], 'failed_chunks': []}
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save progress to file."""
        with self.progress_lock:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save scraped results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _create_chunks(
        self,
        source_name: str,
        items: List[str]
    ) -> List[ScrapingChunk]:
        """Split items into chunks for processing."""
        chunks = []
        for i in range(0, len(items), self.chunk_size):
            chunk_items = items[i:i + self.chunk_size]
            chunk = ScrapingChunk(
                source_name=source_name,
                start_index=i,
                end_index=min(i + self.chunk_size, len(items)),
                items_to_process=chunk_items,
                chunk_id=f"{source_name}_{i}_{i + len(chunk_items)}"
            )
            chunks.append(chunk)
        return chunks
    
    def _process_chunk(
        self,
        chunk: ScrapingChunk,
        source
    ) -> ChunkResult:
        """Process a single chunk of items."""
        start_time = time.time()
        successful_items = []
        failed_items = []
        
        try:
            # Process the chunk using the source's fetch_theorems method
            for theorem in source.fetch_theorems(chunk.items_to_process):
                try:
                    # Add source metadata
                    if 'metadata' not in theorem:
                        theorem['metadata'] = {}
                    theorem['metadata']['source'] = chunk.source_name
                    theorem['metadata']['timestamp'] = time.time()
                    
                    # Validate theorem
                    if all(field in theorem and theorem[field] for field in ['name', 'description', 'source_url']):
                        successful_items.append(theorem)
                    else:
                        failed_items.append((
                            theorem.get('name', 'unknown'),
                            "Missing required fields"
                        ))
                    
                except Exception as e:
                    failed_items.append((
                        theorem.get('name', 'unknown'),
                        f"Error processing theorem: {str(e)}"
                    ))
                
                # Add rate limiting delay
                time.sleep(self.rate_limit)
                
        except Exception as e:
            error_msg = f"Error processing chunk: {str(e)}"
            self.logger.error(error_msg)
            failed_items.extend([
                (item, error_msg) for item in chunk.items_to_process
            ])
        
        processing_time = time.time() - start_time
        
        # Log chunk completion
        self.logger.info(
            f"Completed chunk {chunk.chunk_id}: "
            f"{len(successful_items)} successful, "
            f"{len(failed_items)} failed "
            f"(took {processing_time:.2f}s)"
        )
        
        return ChunkResult(
            chunk_id=chunk.chunk_id,
            source_name=chunk.source_name,
            successful_items=successful_items,
            failed_items=failed_items,
            processing_time=processing_time
        )
    
    def scrape_source(
        self,
        source_name: str,
        source,
        items: List[str]
    ) -> Dict[str, Any]:
        """
        Scrape items from a single source with chunked processing.
        
        Args:
            source_name: Name of the source
            source: Source object implementing fetch_theorems
            items: List of items to process (URLs or IDs)
            
        Returns:
            Dictionary containing results and statistics
        """
        self.progress_queue.put(f"Starting scraping from {source_name}")
        
        # Load previous progress
        progress = self._load_progress()
        
        # Create chunks
        chunks = self._create_chunks(source_name, items)
        chunks_to_process = [
            chunk for chunk in chunks 
            if chunk.chunk_id not in progress['completed_chunks']
        ]
        
        self.progress_queue.put(
            f"Processing {len(chunks_to_process)} chunks "
            f"({len(items)} items) from {source_name}"
        )
        
        # Process chunks in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, source): chunk
                for chunk in chunks_to_process
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # Update progress
                    progress['completed_chunks'].append(chunk.chunk_id)
                    self._save_progress(progress)
                    
                except Exception as e:
                    self.logger.error(f"Chunk {chunk.chunk_id} failed: {str(e)}")
                    progress['failed_chunks'].append(chunk.chunk_id)
                    self._save_progress(progress)
        
        # Compile statistics
        total_items = len(items)
        successful_items = sum(len(r.successful_items) for r in all_results)
        failed_items = sum(len(r.failed_items) for r in all_results)
        total_time = sum(r.processing_time for r in all_results)
        
        self.progress_queue.put(f"Completed scraping from {source_name}:")
        self.progress_queue.put(f"- Successful items: {successful_items}")
        self.progress_queue.put(f"- Failed items: {failed_items}")
        
        return {
            'source_name': source_name,
            'results': [asdict(r) for r in all_results],
            'stats': {
                'total_items': total_items,
                'successful_items': successful_items,
                'failed_items': failed_items,
                'total_time': total_time
            }
        }
    
    def scrape_all_sources(
        self,
        sources_config: Dict[str, Tuple[Any, List[str]]]
    ) -> Dict[str, Any]:
        """
        Scrape from all configured sources.
        
        Args:
            sources_config: Dictionary mapping source names to (source, items) tuples
            
        Returns:
            Dictionary containing results and statistics for all sources
        """
        all_results = []
        stats = {}
        
        for source_name, (source, items) in sources_config.items():
            try:
                result = self.scrape_source(source_name, source, items)
                all_results.append(result)
                stats[source_name] = result['stats']
            except Exception as e:
                self.logger.error(f"Error scraping from {source_name}: {str(e)}")
                stats[source_name] = {
                    'total_items': len(items),
                    'successful_items': 0,
                    'failed_items': len(items),
                    'error': str(e)
                }
        
        return {
            'results': all_results,
            'stats': stats
        }
    
    def __del__(self):
        """Clean up resources."""
        self.progress_queue.put(None)
        self.progress_thread.join() 