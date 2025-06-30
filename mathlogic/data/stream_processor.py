"""
Stream processor for efficient large-scale theorem data processing.
"""

import logging
from typing import Dict, Any, List, Optional, Generator
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import gc
import re
from collections import defaultdict

from .chunk_manager import ChunkManager, DataChunk

class StreamProcessor:
    """
    Processes large amounts of theorem data efficiently using streaming and chunking.
    
    Features:
    - Streaming data processing with minimal memory footprint
    - Parallel processing of independent chunks
    - Progressive relationship building
    - Efficient graph construction
    """
    
    def __init__(
        self,
        output_dir: str = "entailment_output/processed_data",
        chunk_size: int = 1000,
        max_workers: int = 4
    ):
        self.chunk_manager = ChunkManager(output_dir, chunk_size)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe processing queues
        self.processing_queue = Queue()
        self.results_queue = Queue()
        
        # Initialize workers
        self.workers = []
        self._start_workers()
        
        # Initialize relationship patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize patterns for relationship detection."""
        self.patterns = {
            'proves': [
                r'proves\s+(?:that\s+)?([^\.]+)',
                r'proof\s+of\s+([^\.]+)',
                r'implies\s+(?:that\s+)?([^\.]+)',
                r'demonstrates\s+(?:that\s+)?([^\.]+)'
            ],
            'depends_on': [
                r'depends\s+on\s+([^\.]+)',
                r'requires\s+([^\.]+)',
                r'assumes\s+([^\.]+)',
                r'based\s+on\s+([^\.]+)'
            ],
            'equivalent_to': [
                r'equivalent\s+to\s+([^\.]+)',
                r'same\s+as\s+([^\.]+)',
                r'if\s+and\s+only\s+if\s+([^\.]+)'
            ],
            'related_to': [
                r'related\s+to\s+([^\.]+)',
                r'connected\s+with\s+([^\.]+)',
                r'similar\s+to\s+([^\.]+)'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {
            rel_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rel_type, patterns in self.patterns.items()
        }
    
    def _start_workers(self):
        """Start worker threads for parallel processing."""
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._process_queue)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _process_queue(self):
        """Worker thread function for processing items from the queue."""
        while True:
            item = self.processing_queue.get()
            if item is None:
                break
                
            try:
                chunk, processor_func = item
                result = processor_func(chunk)
                self.results_queue.put((chunk.chunk_id, result))
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                self.results_queue.put((chunk.chunk_id, None))
            finally:
                self.processing_queue.task_done()
    
    def process_stream(
        self,
        data_iterator: Generator[Dict[str, Any], None, None],
        source: str
    ) -> None:
        """
        Process a stream of data using the chunk manager.
        
        Args:
            data_iterator: Generator yielding theorem data
            source: Source identifier
        """
        self.chunk_manager.process_stream(data_iterator, source)
    
    def build_relationships_stream(
        self,
        source_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build relationships progressively from stored chunks.
        
        Args:
            source_filter: Optional source to filter by
            
        Returns:
            DataFrame containing relationship information
        """
        relationships = []
        theorem_texts = {}  # Cache theorem texts
        
        # First pass: collect all theorem texts
        for chunk in self.chunk_manager.stream_data(source_filter=source_filter):
            for theorem in chunk.theorems:
                if 'name' in theorem and 'description' in theorem:
                    theorem_texts[theorem['name']] = theorem['description']
        
        # Second pass: detect relationships
        for chunk in self.chunk_manager.stream_data(source_filter=source_filter):
            self.processing_queue.put((
                (chunk, theorem_texts),
                self._process_relationships_enhanced
            ))
        
        # Wait for all processing to complete
        self.processing_queue.join()
        
        # Collect results
        while not self.results_queue.empty():
            chunk_id, result = self.results_queue.get()
            if result is not None:
                relationships.extend(result)
        
        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(relationships)
        if not df.empty:
            df = df.drop_duplicates(subset=['source', 'target', 'type'])
        
        return df
    
    def _process_relationships_enhanced(self, data: tuple) -> List[Dict[str, Any]]:
        """Process relationships with enhanced detection."""
        chunk, theorem_texts = data
        processed = []
        
        for theorem in chunk.theorems:
            if 'name' not in theorem or 'description' not in theorem:
                continue
                
            source_name = theorem['name']
            description = theorem['description']
            
            # Check for explicit relationships
            if 'relationships' in theorem:
                for rel in theorem['relationships']:
                    processed.append({
                        'source': source_name,
                        'target': rel['target_theorem'],
                        'type': rel['relationship_type'],
                        'confidence': rel.get('confidence', 1.0),
                        'source_data': chunk.source,
                        'method': 'explicit'
                    })
            
            # Detect relationships from description
            for rel_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(description)
                    for match in matches:
                        target_name = match.group(1).strip()
                        # Check if target exists in our theorem set
                        if target_name in theorem_texts:
                            processed.append({
                                'source': source_name,
                                'target': target_name,
                                'type': rel_type,
                                'confidence': 0.8,  # Lower confidence for pattern-based detection
                                'source_data': chunk.source,
                                'method': 'pattern'
                            })
            
            # Look for theorem references in metadata
            if 'metadata' in theorem:
                metadata = theorem['metadata']
                if 'related_theorems' in metadata:
                    for related in metadata['related_theorems']:
                        if related in theorem_texts:
                            processed.append({
                                'source': source_name,
                                'target': related,
                                'type': 'related_to',
                                'confidence': 0.7,
                                'source_data': chunk.source,
                                'method': 'metadata'
                            })
        
        return processed
    
    def build_graph_stream(
        self,
        relationship_df: pd.DataFrame,
        chunk_size: Optional[int] = None
    ) -> 'nx.DiGraph':
        """
        Build a graph progressively from relationship data.
        
        Args:
            relationship_df: DataFrame containing relationships
            chunk_size: Optional chunk size for processing
            
        Returns:
            NetworkX DiGraph
        """
        import networkx as nx
        G = nx.DiGraph()
        
        if relationship_df.empty:
            self.logger.warning("No relationships found to build graph")
            return G
        
        # Process relationships in chunks to manage memory
        chunk_size = chunk_size or 1000
        for i in range(0, len(relationship_df), chunk_size):
            chunk = relationship_df.iloc[i:i + chunk_size]
            
            # Add edges from this chunk
            edges = [
                (row['source'], row['target'], {
                    'type': row['type'],
                    'confidence': row['confidence'],
                    'source_data': row['source_data'],
                    'method': row.get('method', 'unknown')
                })
                for _, row in chunk.iterrows()
                if pd.notna(row['source']) and pd.notna(row['target'])
            ]
            G.add_edges_from(edges)
            
            # Monitor memory
            if i % (chunk_size * 10) == 0:
                self.chunk_manager._monitor_memory()
        
        return G
    
    def analyze_stream(
        self,
        graph: 'nx.DiGraph',
        analysis_func,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the graph in a streaming fashion.
        
        Args:
            graph: NetworkX DiGraph to analyze
            analysis_func: Function to apply to each chunk
            chunk_size: Optional chunk size for processing
            
        Returns:
            Dictionary containing analysis results
        """
        chunk_size = chunk_size or 1000
        nodes = list(graph.nodes())
        results = []
        
        # Process nodes in chunks
        for i in range(0, len(nodes), chunk_size):
            chunk_nodes = nodes[i:i + chunk_size]
            subgraph = graph.subgraph(chunk_nodes)
            
            try:
                chunk_result = analysis_func(subgraph)
                results.append(chunk_result)
            except Exception as e:
                self.logger.error(f"Error analyzing chunk {i}: {str(e)}")
            
            # Monitor memory
            if i % (chunk_size * 10) == 0:
                self.chunk_manager._monitor_memory()
        
        # Combine results
        combined_results = defaultdict(list)
        for result in results:
            for key, value in result.items():
                combined_results[key].append(value)
        
        # Aggregate results
        final_results = {}
        for key, values in combined_results.items():
            if isinstance(values[0], (int, float)):
                final_results[key] = np.mean(values)
            else:
                final_results[key] = values
        
        return final_results
    
    def cleanup(self):
        """Clean up resources and stop workers."""
        # Signal all workers to stop
        for _ in self.workers:
            self.processing_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        # Clear queues
        while not self.processing_queue.empty():
            self.processing_queue.get()
        while not self.results_queue.empty():
            self.results_queue.get()
        
        # Force garbage collection
        gc.collect()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()