"""
Pytest configuration and fixtures for mathlogic tests.
"""

import pytest
from mathlogic.core.entailment import EntailmentCone, LogicalStatement, EntailmentRelation

@pytest.fixture
def basic_statements():
    """Fixture providing basic logical statements."""
    return {
        'zfc': LogicalStatement("ZFC", "Zermelo-Fraenkel set theory with Choice", "Set Theory", True),
        'ch': LogicalStatement("CH", "Continuum Hypothesis", "Set Theory", False),
        'ac': LogicalStatement("AC", "Axiom of Choice", "Set Theory", True),
        'pa': LogicalStatement("PA", "Peano Arithmetic", "Number Theory", True),
        'con_pa': LogicalStatement("Con(PA)", "Consistency of PA", "Meta-mathematics", False)
    }

@pytest.fixture
def basic_cone(basic_statements):
    """Fixture providing a basic entailment cone with statements."""
    cone = EntailmentCone()
    
    # Add all statements
    for stmt in basic_statements.values():
        cone.add_statement(stmt)
    
    # Add some basic relations
    relations = [
        EntailmentRelation(basic_statements['zfc'], basic_statements['ac'], "Contains"),
        EntailmentRelation(basic_statements['zfc'], basic_statements['ch'], "Independence"),
        EntailmentRelation(basic_statements['pa'], basic_statements['con_pa'], "Independence")
    ]
    
    for rel in relations:
        cone.add_relation(rel)
    
    return cone

@pytest.fixture
def full_graph(basic_cone):
    """Fixture providing the NetworkX graph from the basic cone."""
    return basic_cone.graph

