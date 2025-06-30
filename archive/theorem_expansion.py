"""
Additional theorems and independence results for the entailment graph.
This file contains a comprehensive collection of mathematical theorems,
their relationships, and independence results.
"""

ADDITIONAL_THEOREMS = {
    'Gödel Incompleteness Theorems': {
        'description': 'No consistent formal system containing basic arithmetic can prove its own consistency',
        'proven_by': ['ZF'],
        'year_proven': 1931,
        'complexity': 'high',
        'related_theorems': ['Second Incompleteness Theorem']
    },
    'Second Incompleteness Theorem': {
        'description': 'No consistent formal system can prove its own consistency',
        'proven_by': ['ZF'],
        'year_proven': 1931,
        'complexity': 'high'
    },
    'Diamond Principle': {
        'description': 'A combinatorial principle independent of ZFC',
        'independent_of': ['ZFC'],
        'proven_independent': 1972,
        'complexity': 'high'
    },
    'Suslin Hypothesis': {
        'description': 'Every complete dense linear order without endpoints that is ccc is isomorphic to the real line',
        'independent_of': ['ZFC'],
        'proven_independent': 1971,
        'complexity': 'high'
    },
    'Whitehead Problem': {
        'description': 'Is every Whitehead group free?',
        'independent_of': ['ZFC'],
        'proven_independent': 1974,
        'complexity': 'high'
    },
    'Large Cardinal Axioms': {
        'description': 'Various axioms asserting existence of very large cardinals',
        'independent_of': ['ZFC'],
        'complexity': 'high',
        'subtypes': ['Measurable Cardinals', 'Woodin Cardinals', 'Supercompact Cardinals']
    },
    'Paris-Harrington Principle': {
        'description': 'A true combinatorial statement unprovable in Peano Arithmetic',
        'proven_by': ['ZFC'],
        'independent_of': ['PA'],
        'year_proven': 1977,
        'complexity': 'high'
    },
    'Goodstein Theorem': {
        'description': 'Every Goodstein sequence eventually terminates at 0',
        'proven_by': ['ZFC'],
        'independent_of': ['PA'],
        'year_proven': 1944,
        'complexity': 'medium'
    },
    'Borel Determinacy Theorem': {
        'description': 'Every Borel game is determined',
        'proven_by': ['ZFC'],
        'year_proven': 1975,
        'complexity': 'high'
    },
    'Projective Determinacy': {
        'description': 'Every projective game is determined',
        'independent_of': ['ZFC'],
        'proven_by': ['ZFC+LC'],
        'complexity': 'high'
    },
    'Kruskal Theorem': {
        'description': 'A theorem about well-quasi-orderings of finite trees',
        'proven_by': ['ZFC'],
        'independent_of': ['ATR0'],
        'year_proven': 1960,
        'complexity': 'medium'
    },
    'Generalized Continuum Hypothesis': {
        'description': 'For any infinite cardinal κ, there is no cardinal λ such that κ < λ < 2^κ',
        'independent_of': ['ZFC'],
        'proven_independent': 1963,
        'complexity': 'high'
    }
}

# Additional formal systems
ADDITIONAL_SYSTEMS = {
    'ATR0': {
        'description': 'Arithmetical Transfinite Recursion',
        'contains': ['PA'],
        'strength': 0.75
    },
    'ZFC+PFA': {
        'description': 'ZFC with Proper Forcing Axiom',
        'contains': ['ZFC'],
        'strength': 1.2
    },
    'ZFC+AD': {
        'description': 'ZFC with Axiom of Determinacy',
        'contains': ['ZFC'],
        'incompatible_with': ['AC'],
        'strength': 1.3
    }
}

# Independence relationships between theorems
INDEPENDENCE_RELATIONSHIPS = [
    ('Continuum Hypothesis', 'Generalized Continuum Hypothesis', 'implies'),
    ('Diamond Principle', 'Suslin Hypothesis', 'independent'),
    ('Large Cardinal Axioms', 'Projective Determinacy', 'implies'),
    ('Paris-Harrington Principle', 'Goodstein Theorem', 'independent'),
    ('ZFC+AD', 'Projective Determinacy', 'proves'),
    ('Gödel Incompleteness Theorems', 'Second Incompleteness Theorem', 'implies')
]

def update_known_theorems():
    """Update the KNOWN_THEOREMS dictionary with additional theorems."""
    from data_validation import KNOWN_THEOREMS
    KNOWN_THEOREMS.update(ADDITIONAL_THEOREMS)
    return KNOWN_THEOREMS

def update_formal_systems():
    """Update the FORMAL_SYSTEMS dictionary with additional systems."""
    from data_validation import FORMAL_SYSTEMS
    FORMAL_SYSTEMS.update(ADDITIONAL_SYSTEMS)
    return FORMAL_SYSTEMS

def get_independence_relationships():
    """Get the list of independence relationships between theorems."""
    return INDEPENDENCE_RELATIONSHIPS 