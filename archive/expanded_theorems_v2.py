"""
Expanded collection of mathematical theorems, independence results, and their relationships.
This file contains a comprehensive database of mathematical theorems across various fields.
Version 2.0 with significantly expanded coverage.
"""

from typing import Dict, List, Tuple, Set

# Algebraic Geometry Theorems
ALGEBRAIC_GEOMETRY_THEOREMS = {
    'Mordell Conjecture': {
        'description': 'For any curve of genus greater than 1 defined over a number field, the set of rational points is finite',
        'proven_by': ['ZFC'],
        'year_proven': 1983,
        'prover': 'Gerd Faltings',
        'complexity': 'high',
        'field': 'algebraic_geometry'
    },
    'Resolution of Singularities': {
        'description': 'Any algebraic variety over a field of characteristic zero admits a resolution of singularities',
        'proven_by': ['ZFC'],
        'year_proven': 1964,
        'prover': 'Heisuke Hironaka',
        'complexity': 'high',
        'field': 'algebraic_geometry'
    },
    'Weil Conjectures': {
        'description': 'Deep relationships between topology and number theory for varieties over finite fields',
        'proven_by': ['ZFC'],
        'year_proven': 1974,
        'prover': 'Pierre Deligne',
        'complexity': 'high',
        'field': 'algebraic_geometry'
    }
}

# Differential Geometry Theorems
DIFFERENTIAL_GEOMETRY_THEOREMS = {
    'Atiyah-Singer Index Theorem': {
        'description': 'Relates the analytic index of an elliptic differential operator to topological data',
        'proven_by': ['ZFC'],
        'year_proven': 1963,
        'complexity': 'high',
        'field': 'differential_geometry'
    },
    'Thurston Geometrization Conjecture': {
        'description': 'Every closed 3-manifold can be decomposed into geometric pieces',
        'proven_by': ['ZFC'],
        'year_proven': 2003,
        'prover': 'Grigori Perelman',
        'complexity': 'high',
        'field': 'differential_geometry'
    },
    'Positive Mass Theorem': {
        'description': 'The total mass of an asymptotically flat manifold with positive scalar curvature is positive',
        'proven_by': ['ZFC'],
        'year_proven': 1979,
        'complexity': 'high',
        'field': 'differential_geometry'
    }
}

# Model Theory Theorems
MODEL_THEORY_THEOREMS = {
    'Morley Theorem': {
        'description': 'A first-order theory categorical in one uncountable cardinal is categorical in all uncountable cardinals',
        'proven_by': ['ZFC'],
        'year_proven': 1965,
        'complexity': 'high',
        'field': 'model_theory'
    },
    'Shelah Classification Theory': {
        'description': 'Classification of first-order theories via stability-theoretic properties',
        'proven_by': ['ZFC'],
        'year_proven': 1978,
        'complexity': 'high',
        'field': 'model_theory'
    },
    'Keisler-Shelah Isomorphism Theorem': {
        'description': 'Two elementarily equivalent structures have isomorphic ultrapowers',
        'proven_by': ['ZFC'],
        'year_proven': 1964,
        'complexity': 'medium',
        'field': 'model_theory'
    }
}

# Proof Theory Theorems
PROOF_THEORY_THEOREMS = {
    'Gentzen Consistency Proof': {
        'description': 'Consistency of PA proved using transfinite induction up to ε₀',
        'proven_by': ['ZFC'],
        'year_proven': 1936,
        'complexity': 'high',
        'field': 'proof_theory'
    },
    'Takeuti Fundamental Conjecture': {
        'description': 'Cut-elimination theorem for higher-order logic',
        'proven_by': ['ZFC'],
        'year_proven': 1978,
        'complexity': 'high',
        'field': 'proof_theory'
    },
    'Π¹₁-Comprehension Independence': {
        'description': 'Independence of Π¹₁-comprehension from predicative analysis',
        'proven_by': ['ZFC'],
        'year_proven': 1967,
        'complexity': 'high',
        'field': 'proof_theory'
    }
}

# Algebraic Topology Theorems
ALGEBRAIC_TOPOLOGY_THEOREMS = {
    'Adams Conjecture': {
        'description': 'Relationship between vector bundles and K-theory operations',
        'proven_by': ['ZFC'],
        'year_proven': 1970,
        'complexity': 'high',
        'field': 'algebraic_topology'
    },
    'Quillen-Suslin Theorem': {
        'description': 'Projective modules over polynomial rings are free',
        'proven_by': ['ZFC'],
        'year_proven': 1976,
        'complexity': 'high',
        'field': 'algebraic_topology'
    },
    'Bloch-Kato Conjecture': {
        'description': 'Relationship between Milnor K-theory and Galois cohomology',
        'proven_by': ['ZFC'],
        'year_proven': 2011,
        'complexity': 'high',
        'field': 'algebraic_topology'
    }
}

# Homological Algebra Theorems
HOMOLOGICAL_ALGEBRA_THEOREMS = {
    'Mitchell Embedding Theorem': {
        'description': 'Every small abelian category fully embeds in a category of modules',
        'proven_by': ['ZFC'],
        'year_proven': 1964,
        'complexity': 'medium',
        'field': 'homological_algebra'
    },
    'Grothendieck Spectral Sequence': {
        'description': 'Spectral sequence relating derived functors of composite functors',
        'proven_by': ['ZFC'],
        'year_proven': 1957,
        'complexity': 'high',
        'field': 'homological_algebra'
    },
    'Ext-Tor Duality': {
        'description': 'Duality between Ext and Tor functors',
        'proven_by': ['ZFC'],
        'year_proven': 1956,
        'complexity': 'medium',
        'field': 'homological_algebra'
    }
}

# Descriptive Set Theory Theorems
DESCRIPTIVE_SET_THEORY_THEOREMS = {
    'Perfect Set Property': {
        'description': 'Every uncountable analytic set contains a perfect subset',
        'proven_by': ['ZFC'],
        'year_proven': 1916,
        'complexity': 'medium',
        'field': 'descriptive_set_theory'
    },
    'Determinacy of Analytic Games': {
        'description': 'All analytic games are determined',
        'proven_by': ['ZFC'],
        'year_proven': 1975,
        'complexity': 'high',
        'field': 'descriptive_set_theory'
    },
    'Mansfield-Solovay Theorem': {
        'description': 'Characterization of sets constructible from a real',
        'proven_by': ['ZFC'],
        'year_proven': 1970,
        'complexity': 'high',
        'field': 'descriptive_set_theory'
    }
}

# Recursion Theory Theorems
RECURSION_THEORY_THEOREMS = {
    'Friedberg-Muchnik Theorem': {
        'description': 'Existence of incomparable recursively enumerable degrees',
        'proven_by': ['PA'],
        'year_proven': 1956,
        'complexity': 'medium',
        'field': 'recursion_theory'
    },
    'Sacks Density Theorem': {
        'description': 'The Turing degrees are dense',
        'proven_by': ['PA'],
        'year_proven': 1964,
        'complexity': 'high',
        'field': 'recursion_theory'
    },
    'Shore-Slaman Join Theorem': {
        'description': 'Characterization of joins in the Turing degrees',
        'proven_by': ['PA'],
        'year_proven': 1999,
        'complexity': 'high',
        'field': 'recursion_theory'
    }
}

# Additional Formal Systems
EXPANDED_SYSTEMS_V2 = {
    'ZFC+I3': {
        'description': 'ZFC with third-order large cardinal axiom',
        'contains': ['ZFC+I2', 'ZFC+I1', 'ZFC+I0'],
        'strength': 1.6
    },
    'ZFC+Woodin': {
        'description': 'ZFC with existence of Woodin cardinals',
        'contains': ['ZFC+MM'],
        'strength': 1.7
    },
    'WKL0': {
        'description': 'Weak König\'s Lemma',
        'contains': ['RCA0'],
        'strength': 0.4
    },
    'ACA0₀': {
        'description': 'Arithmetical Comprehension restricted to Σ⁰₁ formulas',
        'contains': ['RCA0'],
        'strength': 0.5
    },
    'CZF': {
        'description': 'Constructive Zermelo-Fraenkel set theory',
        'contains': ['IZF'],
        'strength': 0.8
    },
    'MLᵢ': {
        'description': 'Martin-Löf type theory with universes',
        'contains': ['HOL'],
        'strength': 0.9
    },
    'NFU+Con(NFU)': {
        'description': 'New Foundations with Urelements plus its own consistency',
        'contains': ['NFU'],
        'strength': 0.75
    },
    'TST': {
        'description': 'Theory of Simple Types',
        'contains': [],
        'strength': 0.6
    }
}

# Independence and Implication Relationships
EXPANDED_RELATIONSHIPS_V2 = [
    ('ZFC+I3', 'ZFC+I2', 'contains'),
    ('ZFC+Woodin', 'ZFC+MM', 'contains'),
    ('WKL0', 'RCA0', 'contains'),
    ('ACA0₀', 'RCA0', 'contains'),
    ('CZF', 'IZF', 'contains'),
    ('MLᵢ', 'HOL', 'contains'),
    ('NFU+Con(NFU)', 'NFU', 'contains'),
    ('Morley Theorem', 'Shelah Classification Theory', 'implies'),
    ('Gentzen Consistency Proof', 'Con(PA)', 'proves'),
    ('Determinacy of Analytic Games', 'Perfect Set Property', 'implies'),
    ('Shore-Slaman Join Theorem', 'Sacks Density Theorem', 'implies'),
    ('Resolution of Singularities', 'Hironaka Desingularization', 'proves'),
    ('Thurston Geometrization Conjecture', 'Poincaré Conjecture', 'implies'),
    ('Bloch-Kato Conjecture', 'Quillen-Suslin Theorem', 'independent'),
    ('ZFC', 'Determinacy of Analytic Games', 'proves'),
    ('PA', 'Friedberg-Muchnik Theorem', 'proves'),
    ('ZFC+Woodin', 'Projective Determinacy', 'proves')
]

def get_all_theorems_v2() -> Dict:
    """Combine all theorem collections into one dictionary."""
    all_theorems = {}
    all_theorems.update(ALGEBRAIC_GEOMETRY_THEOREMS)
    all_theorems.update(DIFFERENTIAL_GEOMETRY_THEOREMS)
    all_theorems.update(MODEL_THEORY_THEOREMS)
    all_theorems.update(PROOF_THEORY_THEOREMS)
    all_theorems.update(ALGEBRAIC_TOPOLOGY_THEOREMS)
    all_theorems.update(HOMOLOGICAL_ALGEBRA_THEOREMS)
    all_theorems.update(DESCRIPTIVE_SET_THEORY_THEOREMS)
    all_theorems.update(RECURSION_THEORY_THEOREMS)
    return all_theorems

def get_all_systems_v2() -> Dict:
    """Get all formal systems."""
    return EXPANDED_SYSTEMS_V2

def get_all_relationships_v2() -> List[Tuple[str, str, str]]:
    """Get all relationships between theorems."""
    return EXPANDED_RELATIONSHIPS_V2

def get_theorems_by_field_v2(field: str) -> Dict:
    """Get all theorems in a specific field."""
    all_theorems = get_all_theorems_v2()
    return {k: v for k, v in all_theorems.items() if v.get('field') == field}

def get_independent_theorems_v2() -> List[str]:
    """Get list of theorems known to be independent of their base systems."""
    return [name for name, data in get_all_theorems_v2().items() 
            if 'independent_of' in data or data.get('status') == 'open'] 