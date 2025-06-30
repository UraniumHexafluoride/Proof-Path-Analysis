"""
Tests for core functionality of the mathlogic package.
"""

import pytest
import networkx as nx
from mathlogic.core.entailment import EntailmentCone, LogicalStatement, EntailmentRelation

def test_logical_statement_creation():
    """Test creation of logical statements."""
    stmt = LogicalStatement("ZFC", "Zermelo-Fraenkel set theory with Choice", "Set Theory", True)
    assert stmt.symbol == "ZFC"
    assert stmt.description == "Zermelo-Fraenkel set theory with Choice"
    assert stmt.formal_system == "Set Theory"
    assert stmt.is_axiom is True

def test_logical_statement_equality():
    """Test logical statement equality comparison."""
    stmt1 = LogicalStatement("ZFC", "First description")
    stmt2 = LogicalStatement("ZFC", "Second description")
    stmt3 = LogicalStatement("PA", "Different statement")
    
    assert stmt1 == stmt2  # Same symbol means same statement
    assert stmt1 != stmt3  # Different symbols mean different statements
    assert hash(stmt1) == hash(stmt2)  # Hash should be based on symbol

def test_entailment_cone_creation(basic_statements):
    """Test creation of entailment cone."""
    cone = EntailmentCone()
    assert isinstance(cone.graph, nx.DiGraph)
    assert len(cone.statements) == 0
    assert len(cone.relations) == 0
    
    # Add a statement
    cone.add_statement(basic_statements['zfc'])
    assert len(cone.statements) == 1
    assert len(cone.graph.nodes()) == 1

def test_adding_relations(basic_cone, basic_statements):
    """Test adding relations to the entailment cone."""
    # Test existing relation
    zfc = basic_statements['zfc']
    ac = basic_statements['ac']
    assert basic_cone.graph.has_edge(zfc.symbol, ac.symbol)
    assert basic_cone.graph.edges[zfc.symbol, ac.symbol]['relation_type'] == "Contains"

def test_invalid_relation(basic_cone, basic_statements):
    """Test adding invalid relations raises appropriate errors."""
    zfc = basic_statements['zfc']
    
    # Test self-relation
    with pytest.raises(ValueError):
        relation = EntailmentRelation(zfc, zfc, "Contains")
        basic_cone.add_relation(relation)
    
    # Test invalid relation type
    with pytest.raises(ValueError):
        relation = EntailmentRelation(zfc, basic_statements['ch'], "InvalidType")
        basic_cone.add_relation(relation)

def test_closure_properties(basic_cone):
    """Test checking of closure properties."""
    properties = basic_cone.check_closure_properties()
    assert isinstance(properties, dict)
    assert all(isinstance(v, bool) or v is None for v in properties.values())
    assert 'transitivity' in properties
    assert 'reflexivity' in properties
    assert 'consistency' in properties

def test_logical_distance(basic_cone, basic_statements):
    """Test computation of logical distance between statements."""
    zfc = basic_statements['zfc']
    ac = basic_statements['ac']
    ch = basic_statements['ch']
    
    # Direct relation
    assert basic_cone.compute_logical_distance(zfc.symbol, ac.symbol) == 1
    
    # No path
    assert basic_cone.compute_logical_distance(ac.symbol, ch.symbol) == float('inf')

def test_graph_properties(basic_cone):
    """Test graph structural properties."""
    graph = basic_cone.graph
    
    # Test node attributes
    for node in graph.nodes():
        attrs = graph.nodes[node]
        assert 'description' in attrs
        assert 'formal_system' in attrs
        assert 'is_axiom' in attrs
    
    # Test edge attributes
    for _, _, attrs in graph.edges(data=True):
        assert 'relation_type' in attrs

def test_adding_duplicate_statement(basic_cone, basic_statements):
    """Test adding duplicate statements."""
    # Create a new statement with same symbol
    duplicate = LogicalStatement("ZFC", "Different description", "Different system", False)
    
    # Adding duplicate should update attributes but not create new node
    initial_node_count = len(basic_cone.graph.nodes())
    basic_cone.add_statement(duplicate)
    assert len(basic_cone.graph.nodes()) == initial_node_count
    
    # Check that attributes were updated
    node_attrs = basic_cone.graph.nodes[duplicate.symbol]
    assert node_attrs['description'] == duplicate.description
    assert node_attrs['formal_system'] == duplicate.formal_system

