"""
Efficient data chunking and streaming manager for large-scale theorem processing.
"""

import logging
from typing import Iterator, Dict, Any, List, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import h5py
import pandas as pd
from tqdm import tqdm
import psutil
import gc

@dataclass
class DataChunk:
    """Represents a chunk of processed data."""
    chunk_id: str
    source: str
    theorems: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]]

class ChunkManager:
    """
    Manages efficient data chunking and streaming for large-scale theorem processing.
    
    Features:
    - Memory-efficient streaming processing
    - Automatic chunk size adjustment based on system memory
    - HDF5-based storage for efficient data access
    - Progressive data loading and processing
    - Memory monitoring and garbage collection
    """
    
    def __init__(
        self,
        output_dir: str = "entailment_output/processed_data",
        initial_chunk_size: int = 1000,
        memory_threshold: float = 0.75  # 75% of available memory
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = initial_chunk_size
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize HDF5 storage
        self.storage_file = self.output_dir / "theorem_data.h5"
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize HDF5 storage structure."""
        try:
            # Delete existing file if it exists
            if self.storage_file.exists():
                self.storage_file.unlink()
            
            with h5py.File(self.storage_file, 'w') as f:
                f.create_group('theorems')
                f.create_group('relationships')
                f.create_group('metadata')
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise
    
    def _monitor_memory(self) -> float:
        """Monitor system memory usage and adjust chunk size if needed."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        # Adjust chunk size based on memory pressure
        if usage_percent > self.memory_threshold:
            self.chunk_size = max(100, self.chunk_size // 2)
            self.logger.warning(
                f"High memory usage ({usage_percent:.2%}). "
                f"Reducing chunk size to {self.chunk_size}"
            )
            gc.collect()  # Force garbage collection
        
        return usage_percent
    
    def process_stream(
        self,
        data_iterator: Iterator[Dict[str, Any]],
        source: str
    ) -> None:
        """
        Process a stream of data in chunks.
        
        Args:
            data_iterator: Iterator yielding theorem data
            source: Source identifier
        """
        current_chunk = []
        chunk_counter = 0
        
        try:
            with h5py.File(self.storage_file, 'a') as f:
                theorems_group = f['theorems']
                relationships_group = f['relationships']
                metadata_group = f['metadata']
                
                for item in tqdm(data_iterator, desc=f"Processing {source}"):
                    if not item:  # Skip empty items
                        continue
                        
                    current_chunk.append(item)
                    
                    # Process chunk when it reaches the target size
                    if len(current_chunk) >= self.chunk_size:
                        self._process_and_store_chunk(
                            current_chunk,
                            source,
                            chunk_counter,
                            theorems_group,
                            relationships_group,
                            metadata_group
                        )
                        current_chunk = []
                        chunk_counter += 1
                        
                        # Monitor memory usage
                        self._monitor_memory()
                
                # Process remaining items
                if current_chunk:
                    self._process_and_store_chunk(
                        current_chunk,
                        source,
                        chunk_counter,
                        theorems_group,
                        relationships_group,
                        metadata_group
                    )
        except Exception as e:
            self.logger.error(f"Error processing stream: {str(e)}")
            # Save failed items for later processing
            failed_path = self.output_dir / f"failed_{source}_stream.json"
            with open(failed_path, 'w') as f:
                json.dump(current_chunk, f)
    
    def _process_and_store_chunk(
        self,
        items: List[Dict[str, Any]],
        source: str,
        chunk_id: int,
        theorems_group: h5py.Group,
        relationships_group: h5py.Group,
        metadata_group: h5py.Group
    ) -> None:
        """Process and store a chunk of data."""
        try:
            # Create unique chunk identifier with timestamp
            chunk_name = f"{source}_{chunk_id}_{int(datetime.now().timestamp())}"
            
            # Extract and store theorems
            theorems_data = pd.DataFrame([
                {k: v for k, v in item.items() if k != 'relationships'}
                for item in items if item  # Skip empty items
            ]).to_json()
            
            # Delete existing dataset if it exists
            if chunk_name in theorems_group:
                del theorems_group[chunk_name]
            theorems_group.create_dataset(
                chunk_name,
                data=theorems_data.encode('utf-8')
            )
            
            # Extract and store relationships
            relationships = []
            for item in items:
                if item and 'relationships' in item:
                    relationships.extend(item['relationships'])
            
            relationships_data = pd.DataFrame(relationships).to_json()
            if chunk_name in relationships_group:
                del relationships_group[chunk_name]
            relationships_group.create_dataset(
                chunk_name,
                data=relationships_data.encode('utf-8')
            )
            
            # Store metadata
            metadata = {
                'source': source,
                'chunk_id': chunk_id,
                'item_count': len(items),
                'timestamp': datetime.now().isoformat()
            }
            if chunk_name in metadata_group:
                del metadata_group[chunk_name]
            metadata_group.create_dataset(
                chunk_name,
                data=json.dumps(metadata).encode('utf-8')
            )
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            # Save failed items for later processing
            failed_path = self.output_dir / f"failed_{source}_{chunk_id}.json"
            with open(failed_path, 'w') as f:
                json.dump(items, f)
    
    def stream_data(
        self,
        chunk_size: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> Iterator[DataChunk]:
        """
        Stream data from storage in manageable chunks.
        
        Args:
            chunk_size: Optional custom chunk size for reading
            source_filter: Optional source to filter by
            
        Yields:
            DataChunk objects containing processed data
        """
        chunk_size = chunk_size or self.chunk_size
        
        try:
            with h5py.File(self.storage_file, 'r') as f:
                theorems_group = f['theorems']
                relationships_group = f['relationships']
                metadata_group = f['metadata']
                
                # Get all chunk names
                chunk_names = list(theorems_group.keys())
                
                if source_filter:
                    chunk_names = [
                        name for name in chunk_names
                        if name.startswith(source_filter)
                    ]
                
                for i in range(0, len(chunk_names), chunk_size):
                    batch_names = chunk_names[i:i + chunk_size]
                    
                    for name in batch_names:
                        try:
                            # Load data for this chunk
                            theorems_data = pd.read_json(
                                theorems_group[name][()].decode('utf-8')
                            ).to_dict('records')
                            
                            relationships_data = pd.read_json(
                                relationships_group[name][()].decode('utf-8')
                            ).to_dict('records')
                            
                            metadata = json.loads(
                                metadata_group[name][()].decode('utf-8')
                            )
                            
                            yield DataChunk(
                                chunk_id=name,
                                source=metadata['source'],
                                theorems=theorems_data,
                                metadata=metadata,
                                relationships=relationships_data
                            )
                        except Exception as e:
                            self.logger.error(f"Error streaming chunk {name}: {str(e)}")
                            continue
                    
                    # Monitor memory after each batch
                    self._monitor_memory()
        except Exception as e:
            self.logger.error(f"Error streaming data: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored data."""
        stats = {
            'total_chunks': 0,
            'total_theorems': 0,
            'total_relationships': 0,
            'sources': set(),
            'storage_size': 0
        }
        
        try:
            with h5py.File(self.storage_file, 'r') as f:
                # Count chunks
                stats['total_chunks'] = len(f['theorems'])
                
                # Gather statistics
                for name in f['theorems']:
                    try:
                        theorems_data = pd.read_json(
                            f['theorems'][name][()].decode('utf-8')
                        )
                        relationships_data = pd.read_json(
                            f['relationships'][name][()].decode('utf-8')
                        )
                        metadata = json.loads(
                            f['metadata'][name][()].decode('utf-8')
                        )
                        
                        stats['total_theorems'] += len(theorems_data)
                        stats['total_relationships'] += len(relationships_data)
                        stats['sources'].add(metadata['source'])
                    except Exception as e:
                        self.logger.error(f"Error processing statistics for chunk {name}: {str(e)}")
                        continue
                
                # Get file size
                stats['storage_size'] = self.storage_file.stat().st_size
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
        
        return stats
    
    def cleanup_old_chunks(self, max_age_days: int = 30) -> None:
        """Clean up chunks older than the specified age."""
        try:
            current_time = datetime.now().timestamp()
            with h5py.File(self.storage_file, 'a') as f:
                for group_name in ['theorems', 'relationships', 'metadata']:
                    group = f[group_name]
                    for chunk_name in list(group.keys()):
                        try:
                            # Extract timestamp from chunk name
                            timestamp = int(chunk_name.split('_')[-1])
                            age_days = (current_time - timestamp) / (24 * 3600)
                            
                            if age_days > max_age_days:
                                del group[chunk_name]
                        except Exception as e:
                            self.logger.error(f"Error cleaning up chunk {chunk_name}: {str(e)}")
                            continue
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")