"""
Core components for the mathlogic package.

This module provides the fundamental data structures for representing
logical statements and their relationships.
"""

from mathlogic.core.entailment import LogicalStatement, EntailmentRelation, EntailmentCone
from mathlogic.core.statements import get_all_theorems, get_all_systems, get_all_relationships

__all__ = [
    'LogicalStatement',
    'EntailmentRelation',
    'EntailmentCone',
    'get_all_theorems',
    'get_all_systems',
    'get_all_relationships'
]

"""
Core components for mathematical logic analysis.
"""

from mathlogic.core.entailment import LogicalStatement, EntailmentRelation, EntailmentCone

__all__ = [
    'LogicalStatement',
    'EntailmentRelation', 
    'EntailmentCone'
]

