"""
Tests for the chunked processing system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import tempfile
import shutil

from mathlogic.data.chunk_manager import ChunkManager, DataChunk
from mathlogic.data.stream_processor import StreamProcessor

def generate_test_data(n_theorems: int = 100) -> list:
    """Generate test theorem data."""
    theorems = []
    for i in range(n_theorems):
        theorem = {
            'name': f'Theorem_{i}',
            'description': f'Description of theorem {i}',
            'source_url': f'http://test.com/theorem_{i}',
            'metadata': {
                'source': 'test',
                'category': f'category_{i % 5}'
            },
            'relationships': [
                {
                    'source_theorem': f'Theorem_{i}',
                    'target_theorem': f'Theorem_{(i + 1) % n_theorems}',
                    'relationship_type': 'implies',
                    'confidence': 0.9
                }
            ]
        }
        theorems.append(theorem)
    return theorems

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_chunk_manager_basic(temp_output_dir):
    """Test basic chunk manager functionality."""
    manager = ChunkManager(temp_output_dir)
    
    # Test data storage and retrieval
    test_data = generate_test_data(100)
    for item in test_data:
        manager.process_stream(iter([item]), 'test')
    
    # Verify data was stored
    stats = manager.get_statistics()
    assert stats['total_theorems'] > 0
    assert 'test' in stats['sources']

def test_stream_processor_graph_building(temp_output_dir):
    """Test graph building with stream processor."""
    processor = StreamProcessor(temp_output_dir)
    
    # Process test data
    test_data = generate_test_data(100)
    processor.process_stream(iter(test_data), 'test')
    
    # Build relationships
    relationships_df = processor.build_relationships_stream()
    assert len(relationships_df) > 0
    
    # Build graph
    graph = processor.build_graph_stream(relationships_df)
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes()) > 0
    assert len(graph.edges()) > 0

def test_memory_efficiency(temp_output_dir):
    """Test memory efficiency with large datasets."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process a larger dataset
    processor = StreamProcessor(temp_output_dir)
    test_data = generate_test_data(1000)  # Larger dataset
    processor.process_stream(iter(test_data), 'test')
    
    # Check memory usage hasn't grown too much
    current_memory = process.memory_info().rss
    memory_increase = (current_memory - initial_memory) / initial_memory
    assert memory_increase < 2.0  # Memory shouldn't more than double

def test_integration_with_structural_analysis(temp_output_dir):
    """Test integration with structural analysis code."""
    processor = StreamProcessor(temp_output_dir)
    
    # Process test data
    test_data = generate_test_data(100)
    processor.process_stream(iter(test_data), 'test')
    
    # Build graph
    relationships_df = processor.build_relationships_stream()
    graph = processor.build_graph_stream(relationships_df)
    
    # Test analysis function
    def analysis_func(g):
        return {
            'n_nodes': len(g.nodes()),
            'n_edges': len(g.edges()),
            'density': nx.density(g),
            'avg_degree': sum(dict(g.degree()).values()) / len(g.nodes())
        }
    
    # Run analysis
    results = processor.analyze_stream(graph, analysis_func)
    
    # Verify results
    assert 'n_nodes' in results
    assert 'n_edges' in results
    assert 'density' in results
    assert 'avg_degree' in results

def test_error_handling(temp_output_dir):
    """Test error handling in chunked processing."""
    processor = StreamProcessor(temp_output_dir)
    
    # Generate data with some invalid entries
    test_data = generate_test_data(100)
    test_data.append({'invalid': 'data'})  # Add invalid data
    
    # Process should continue despite errors
    processor.process_stream(iter(test_data), 'test')
    relationships_df = processor.build_relationships_stream()
    
    # Should have processed valid data
    assert len(relationships_df) > 0

def test_resumability(temp_output_dir):
    """Test processing can be resumed after interruption."""
    manager = ChunkManager(temp_output_dir)
    
    # Process first half of data
    test_data = generate_test_data(100)
    first_half = test_data[:50]
    for item in first_half:
        manager.process_stream(iter([item]), 'test')
    
    # Get initial stats
    initial_stats = manager.get_statistics()
    
    # Process second half
    second_half = test_data[50:]
    for item in second_half:
        manager.process_stream(iter([item]), 'test')
    
    # Get final stats
    final_stats = manager.get_statistics()
    
    # Verify all data was processed
    assert final_stats['total_theorems'] > initial_stats['total_theorems'] 