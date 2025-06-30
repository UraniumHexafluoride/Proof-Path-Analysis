"""
Tests for graph creation functionality of the mathlogic package.
"""

import pytest
import networkx as nx
from mathlogic.graphs.creation import GraphCreator

@pytest.fixture
def basic_data():
    """Fixture providing basic data for graph creation."""
    return {
        'systems': ['ZFC', 'PA', 'ACA0'],
        'theorems': ['CH', 'AC', 'PH'],
        'proves_edges': [('ZFC', 'AC'), ('ACA0', 'PH')],
        'independence_edges': [('ZFC', 'CH'), ('PA', 'PH')],
        'contains_edges': [('ZFC', 'PA'), ('ACA0', 'PA')]
    }

def test_create_entailment_graph(basic_data):
    """Test creation of basic entailment graph."""
    G = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes()) == len(basic_data['systems']) + len(basic_data['theorems'])
    assert len(G.edges()) == (
        len(basic_data['proves_edges']) +
        len(basic_data['independence_edges']) +
        len(basic_data['contains_edges'])
    )

def test_node_attributes(basic_data):
    """Test that nodes have correct attributes."""
    G = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    # Check system nodes
    for system in basic_data['systems']:
        assert G.nodes[system]['type'] == 'system'
    
    # Check theorem nodes
    for theorem in basic_data['theorems']:
        assert G.nodes[theorem]['type'] == 'theorem'

def test_edge_attributes(basic_data):
    """Test that edges have correct attributes."""
    G = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    # Check proves edges
    for source, target in basic_data['proves_edges']:
        assert G.edges[source, target]['relation'] == 'Proves'
    
    # Check independence edges
    for source, target in basic_data['independence_edges']:
        assert G.edges[source, target]['relation'] == 'Independence'
    
    # Check contains edges
    for source, target in basic_data['contains_edges']:
        assert G.edges[source, target]['relation'] == 'Contains'

def test_create_theory_subgraph(basic_data):
    """Test creation of theory subgraph."""
    full_graph = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    subgraph = GraphCreator.create_theory_subgraph(full_graph, 'ZFC', max_depth=2)
    assert isinstance(subgraph, nx.DiGraph)
    assert 'ZFC' in subgraph
    assert 'PA' in subgraph  # Should include contained system
    assert 'AC' in subgraph  # Should include proved theorem

def test_create_independence_graph(basic_data):
    """Test creation of independence graph."""
    full_graph = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    independence_graph = GraphCreator.create_independence_graph(full_graph)
    assert isinstance(independence_graph, nx.Graph)  # Should be undirected
    assert len(independence_graph.edges()) == len(basic_data['independence_edges'])

def test_merge_graphs(basic_data):
    """Test merging of multiple graphs."""
    # Create two separate graphs
    G1 = GraphCreator.create_entailment_graph(
        basic_data['systems'][:1],
        basic_data['theorems'][:1],
        basic_data['proves_edges'][:1],
        basic_data['independence_edges'][:1],
        []
    )
    
    G2 = GraphCreator.create_entailment_graph(
        basic_data['systems'][1:],
        basic_data['theorems'][1:],
        basic_data['proves_edges'][1:],
        basic_data['independence_edges'][1:],
        []
    )
    
    merged = GraphCreator.merge_graphs(G1, G2)
    assert isinstance(merged, nx.DiGraph)
    assert len(merged.nodes()) == len(G1.nodes()) + len(G2.nodes())
    assert len(merged.edges()) == len(G1.edges()) + len(G2.edges())

def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        # Empty system list should raise error
        GraphCreator.create_entailment_graph([], ['CH'], [], [], [])
    
    with pytest.raises(ValueError):
        # Invalid edge (nonexistent node) should raise error
        GraphCreator.create_entailment_graph(
            ['ZFC'], ['CH'],
            [('NonExistent', 'CH')], [], []
        )

def test_subgraph_depth_limit(basic_data):
    """Test depth limiting in theory subgraph creation."""
    full_graph = GraphCreator.create_entailment_graph(
        basic_data['systems'],
        basic_data['theorems'],
        basic_data['proves_edges'],
        basic_data['independence_edges'],
        basic_data['contains_edges']
    )
    
    # Test with depth 1
    subgraph1 = GraphCreator.create_theory_subgraph(full_graph, 'ZFC', max_depth=1)
    assert 'PA' in subgraph1  # Direct connection
    assert 'PH' not in subgraph1  # Indirect connection
    
    # Test with depth 2
    subgraph2 = GraphCreator.create_theory_subgraph(full_graph, 'ZFC', max_depth=2)
    assert 'PH' in subgraph2  # Should now include indirect connection

@pytest.mark.parametrize("input_data", [
    {'systems': ['ZFC'], 'theorems': [], 'proves_edges': [], 'independence_edges': [], 'contains_edges': []},
    {'systems': ['ZFC', 'PA'], 'theorems': ['CH'], 'proves_edges': [], 'independence_edges': [], 'contains_edges': []},
    {'systems': ['ZFC'], 'theorems': ['CH', 'AC'], 'proves_edges': [], 'independence_edges': [('ZFC', 'CH')], 'contains_edges': []}
])
def test_various_graph_configurations(input_data):
    """Test graph creation with various configurations."""
    G = GraphCreator.create_entailment_graph(
        input_data['systems'],
        input_data['theorems'],
        input_data['proves_edges'],
        input_data['independence_edges'],
        input_data['contains_edges']
    )
    
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes()) == len(input_data['systems']) + len(input_data['theorems'])

