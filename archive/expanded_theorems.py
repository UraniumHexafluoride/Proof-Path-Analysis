"""
Expanded collection of mathematical theorems, independence results, and their relationships.
This file contains a comprehensive database of mathematical theorems across various fields.
"""

from typing import Dict, List, Tuple

# Core Set Theory and Logic Theorems
SET_THEORY_THEOREMS = {
    'Cantor Theorem': {
        'description': 'For any set A, the power set P(A) has strictly greater cardinality than A',
        'proven_by': ['ZF'],
        'year_proven': 1874,
        'complexity': 'medium',
        'field': 'set_theory'
    },
    'Zermelo Well-Ordering Theorem': {
        'description': 'Every set can be well-ordered',
        'proven_by': ['ZFC'],
        'equivalent_to': ['AC'],
        'year_proven': 1904,
        'complexity': 'medium',
        'field': 'set_theory'
    },
    'Sierpiński-Erdős Duality': {
        'description': 'CH is equivalent to certain partition properties of uncountable sets',
        'proven_by': ['ZFC'],
        'year_proven': 1943,
        'complexity': 'high',
        'field': 'set_theory'
    },
    'Martin Axiom': {
        'description': 'A weakening of CH that is consistent with both CH and its negation',
        'independent_of': ['ZFC'],
        'proven_independent': 1970,
        'complexity': 'high',
        'field': 'set_theory'
    },
    'Solovay-Tennenbaum Theorem': {
        'description': 'Martin Axiom + negation of CH is consistent with ZFC',
        'proven_by': ['ZFC'],
        'year_proven': 1971,
        'complexity': 'high',
        'field': 'set_theory'
    }
}

# Number Theory and Arithmetic
NUMBER_THEORY_THEOREMS = {
    'Twin Prime Conjecture': {
        'description': 'There are infinitely many pairs of twin primes',
        'status': 'open',
        'complexity': 'high',
        'field': 'number_theory'
    },
    'Goldbach Conjecture': {
        'description': 'Every even integer greater than 2 is the sum of two primes',
        'status': 'open',
        'complexity': 'high',
        'field': 'number_theory'
    },
    'Green-Tao Theorem': {
        'description': 'The primes contain arbitrarily long arithmetic progressions',
        'proven_by': ['ZFC'],
        'year_proven': 2004,
        'complexity': 'high',
        'field': 'number_theory'
    },
    'Erdős-Moser Theorem': {
        'description': 'If the sum of reciprocals of a set of natural numbers diverges, the set contains arbitrarily long arithmetic progressions',
        'proven_by': ['PA'],
        'year_proven': 1953,
        'complexity': 'medium',
        'field': 'number_theory'
    }
}

# Analysis and Topology
ANALYSIS_THEOREMS = {
    'Banach-Tarski Paradox': {
        'description': 'A solid ball can be decomposed and reassembled into two identical copies of itself',
        'proven_by': ['ZFC'],
        'requires': ['AC'],
        'year_proven': 1924,
        'complexity': 'high',
        'field': 'analysis'
    },
    'Vitali Set': {
        'description': 'There exists a subset of R that is not Lebesgue measurable',
        'proven_by': ['ZFC'],
        'requires': ['AC'],
        'year_proven': 1905,
        'complexity': 'medium',
        'field': 'analysis'
    },
    'Borel Hierarchy Theorem': {
        'description': 'The Borel hierarchy does not collapse',
        'proven_by': ['ZFC'],
        'year_proven': 1917,
        'complexity': 'high',
        'field': 'analysis'
    }
}

# Category Theory
CATEGORY_THEOREMS = {
    'Freyd-Mitchell Embedding Theorem': {
        'description': 'Every small abelian category can be embedded in a category of modules',
        'proven_by': ['ZFC'],
        'year_proven': 1964,
        'complexity': 'high',
        'field': 'category_theory'
    },
    'Yoneda Lemma': {
        'description': 'Natural transformations between functors can be completely determined by their components at a single object',
        'proven_by': ['ZFC'],
        'year_proven': 1954,
        'complexity': 'medium',
        'field': 'category_theory'
    }
}

# Additional Formal Systems
EXPANDED_SYSTEMS = {
    'ZFC+MM': {
        'description': 'ZFC with Martin Maximum',
        'contains': ['ZFC', 'MA'],
        'strength': 1.4
    },
    'ZFC+I0': {
        'description': 'ZFC with rank-into-rank axiom',
        'contains': ['ZFC'],
        'strength': 1.5
    },
    'ZF+DC': {
        'description': 'ZF with Dependent Choice',
        'contains': ['ZF'],
        'strength': 0.95
    },
    'NFU': {
        'description': 'New Foundations with Urelements',
        'contains': [],
        'strength': 0.7
    }
}

# Independence and Implication Relationships
EXPANDED_RELATIONSHIPS = [
    ('Martin Axiom', 'Suslin Hypothesis', 'independent'),
    ('AC', 'Banach-Tarski Paradox', 'implies'),
    ('ZFC+MM', 'PFA', 'implies'),
    ('Martin Axiom', 'CH', 'independent'),
    ('Vitali Set', 'AC', 'requires'),
    ('Twin Prime Conjecture', 'Goldbach Conjecture', 'independent'),
    ('Green-Tao Theorem', 'Erdős-Moser Theorem', 'independent'),
    ('Freyd-Mitchell Embedding Theorem', 'Yoneda Lemma', 'independent')
]

def get_all_theorems() -> Dict:
    """Combine all theorem collections into one dictionary."""
    all_theorems = {}
    all_theorems.update(SET_THEORY_THEOREMS)
    all_theorems.update(NUMBER_THEORY_THEOREMS)
    all_theorems.update(ANALYSIS_THEOREMS)
    all_theorems.update(CATEGORY_THEOREMS)
    return all_theorems

def get_all_systems() -> Dict:
    """Get all formal systems."""
    return EXPANDED_SYSTEMS

def get_all_relationships() -> List[Tuple[str, str, str]]:
    """Get all relationships between theorems."""
    return EXPANDED_RELATIONSHIPS

def get_theorems_by_field(field: str) -> Dict:
    """Get all theorems in a specific field."""
    all_theorems = get_all_theorems()
    return {k: v for k, v in all_theorems.items() if v.get('field') == field}

def get_independent_theorems() -> List[str]:
    """Get list of theorems known to be independent of their base systems."""
    return [name for name, data in get_all_theorems().items() 
            if 'independent_of' in data or 'status' == 'open'] 