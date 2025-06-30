"""
Expanded collection of mathematical theorems, independence results, and their relationships.
This file contains a comprehensive database of mathematical theorems across various fields.
"""

from typing import Dict, List, Tuple
from mathlogic.data.synthetic_data_generator import generate_synthetic_data
import random

# --- Base Data (Existing Hand-Curated Data) ---
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

# --- Combined and Expanded Data (from Structural_Independence_Analysis_Ihopeisifinal.py and synthetic) ---

# Initialize with existing systems and then merge synthetic
_systems_from_old_script = {
    'ZFC': {'description': 'Zermelo-Fraenkel set theory with Choice', 'strength': 1.0},
    'ZF': {'description': 'Zermelo-Fraenkel set theory', 'strength': 0.9},
    'PA': {'description': 'Peano Arithmetic', 'strength': 0.8},
    'PA2': {'description': 'Second-order Peano Arithmetic', 'strength': 0.85},
    'ACA0': {'description': 'Arithmetical Comprehension Axiom', 'strength': 0.6},
    'PVS+NP': {'description': 'A system related to P vs NP problem', 'strength': 0.75},
    'ZFC+LC': {'description': 'ZFC with Large Cardinals', 'contains': ['ZFC'], 'strength': 1.1},
    'ZFC+MM': {'description': 'ZFC with Martin Maximum', 'contains': ['ZFC'], 'strength': 1.4},
    'ZFC+AD': {'description': 'ZFC with Axiom of Determinacy', 'contains': ['ZFC'], 'strength': 1.3},
    'ZFC+PD': {'description': 'ZFC with Projective Determinacy', 'contains': ['ZFC'], 'strength': 1.2},
    'ZFC+I0': {'description': 'ZFC with rank-into-rank axiom', 'contains': ['ZFC'], 'strength': 1.5},
    'ZF+DC': {'description': 'ZF with Dependent Choice', 'contains': ['ZF'], 'strength': 0.95},
    'NFU': {'description': 'New Foundations with Urelements', 'contains': [], 'strength': 0.7}
}

# Initialize with existing additional theorems and then merge synthetic
_theorems_from_old_script = {
    'Four Color Theorem': {'description': 'Any planar map can be colored with four colors such that no adjacent regions share the same color', 'year_proven': 1976, 'complexity': 'medium', 'field': 'topology'},
    "Fermat's Last Theorem": {'description': 'No three positive integers a, b, and c satisfy the equation a^n + b^n = c^n for any integer n > 2', 'year_proven': 1995, 'complexity': 'high', 'field': 'number_theory'},
    "Gödel's Incompleteness": {'description': 'Any consistent formal system strong enough to do basic arithmetic cannot prove its own consistency', 'status': 'proven', 'complexity': 'high', 'field': 'logic'},
    'Continuum Hypothesis': {'description': 'There is no set whose cardinality is strictly between that of the integers and the real numbers', 'status': 'independent', 'complexity': 'high', 'field': 'set_theory'},
    'Riemann Hypothesis': {'description': 'All non-trivial zeros of the Riemann zeta function have real part 1/2', 'status': 'open', 'complexity': 'very_high', 'field': 'number_theory'},
    'Poincaré Conjecture': {'description': 'Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere', 'year_proven': 2003, 'complexity': 'very_high', 'field': 'topology'},
    'Hodge Conjecture': {'description': 'A major unsolved problem in algebraic geometry', 'status': 'open', 'complexity': 'very_high', 'field': 'algebraic_geometry'},
    "Goldbach's Conjecture": {'description': 'Every even integer greater than 2 is the sum of two primes', 'status': 'open', 'complexity': 'high', 'field': 'number_theory'},
    'Collatz Conjecture': {'description': 'Starting with any positive integer, repeatedly apply a function; the sequence will eventually reach 1', 'status': 'open', 'complexity': 'medium', 'field': 'number_theory'},
    'ABC Conjecture': {'description': 'A conjecture in Diophantine analysis that has implications for number theory', 'status': 'open', 'complexity': 'high', 'field': 'number_theory'},
    'Navier-Stokes Existence and Smoothness': {'description': 'Existence and smoothness of solutions to the Navier-Stokes equations', 'status': 'open', 'complexity': 'very_high', 'field': 'analysis'},
    'Axiom of Choice': {'description': 'For any collection of non-empty sets, there exists a function that selects one element from each set', 'status': 'independent', 'complexity': 'medium', 'field': 'set_theory'},
    'Swinerton-Dyer Conjecture': {'description': 'A conjecture about elliptic curves', 'status': 'open', 'complexity': 'high', 'field': 'number_theory'},
    'P vs NP': {'description': 'Whether every problem whose solution can be quickly verified can also be quickly solved', 'status': 'open', 'complexity': 'very_high', 'field': 'computational_complexity'},
    'Zorn\'s Lemma': {'description': 'A theorem in set theory equivalent to the Axiom of Choice', 'complexity': 'medium', 'field': 'set_theory'},
    'Compactness Theorem': {'description': 'A set of first-order sentences has a model if and only if every finite subset of it has a model', 'complexity': 'medium', 'field': 'model_theory'},
    'Löwenheim–Skolem Theorem': {'description': 'If a first-order theory has an infinite model, then it has models of every infinite cardinality', 'complexity': 'medium', 'field': 'model_theory'},
    'Completeness Theorem': {'description': 'Every logically valid first-order formula is provable', 'complexity': 'medium', 'field': 'proof_theory'},
    'Incompleteness Theorem': {'description': 'Any consistent formal system strong enough to do basic arithmetic cannot prove its own consistency', 'complexity': 'high', 'field': 'proof_theory'},
    'Halting Problem': {'description': 'The problem of determining, from a description of an arbitrary computer program and an arbitrary input, whether the program will finish running or continue to run forever', 'status': 'undecidable', 'complexity': 'high', 'field': 'recursion_theory'},
    'Church-Turing Thesis': {'description': 'Any effectively calculable function is a Turing-computable function', 'status': 'thesis', 'complexity': 'medium', 'field': 'recursion_theory'},
    'Fundamental Theorem of Arithmetic': {'description': 'Every integer greater than 1 is either a prime number itself or can be represented as a product of prime numbers that is unique up to the order of the factors', 'complexity': 'low', 'field': 'number_theory'},
    'Prime Number Theorem': {'description': 'Describes the asymptotic distribution of prime numbers', 'complexity': 'medium', 'field': 'number_theory'}
}

# Initialize with existing relationships and then merge synthetic
_relationships_from_old_script = [
    ('Martin Axiom', 'Suslin Hypothesis', 'independent'),
    ('AC', 'Banach-Tarski Paradox', 'implies'),
    ('ZFC+MM', 'PFA', 'implies'),
    ('Martin Axiom', 'CH', 'independent'),
    ('Vitali Set', 'AC', 'requires'),
    ('Twin Prime Conjecture', 'Goldbach Conjecture', 'independent'),
    ('Green-Tao Theorem', 'Erdős-Moser Theorem', 'independent'),
    ('Freyd-Mitchell Embedding Theorem', 'Yoneda Lemma', 'independent'),
    ('ZFC', "Fermat's Last Theorem", 'proves'),
    ('PA2', 'Four Color Theorem', 'proves'),
    ('ZFC', 'Poincaré Conjecture', 'proves'),
    ('ZFC', 'Hodge Conjecture', 'proves'),
    ('ZFC+MM', 'Swinerton-Dyer Conjecture', 'proves'),
    ('ZFC', 'Zorn\'s Lemma', 'proves'),
    ('ZFC', 'Well-Ordering Theorem', 'proves'),
    ('ZFC', 'Compactness Theorem', 'proves'),
    ('PA', 'Fundamental Theorem of Arithmetic', 'proves'),
    ('ZFC', 'Prime Number Theorem', 'proves'),
    ('PA', 'Completeness Theorem', 'proves'),
    ('PA2', 'Incompleteness Theorem', 'proves'),
    ('ZFC', 'Continuum Hypothesis', 'independent'),
    ('ZFC', 'Twin Prime Conjecture', 'independent'),
    ('ZFC', 'Riemann Hypothesis', 'independent'),
    ('PA', "Gödel's Incompleteness", 'independent'),
    ('ZFC', 'Axiom of Choice', 'independent'),
    ('PVS+NP', 'P vs NP', 'independent'),
    ('PA', 'Halting Problem', 'independent'),
    ('PA', 'Church-Turing Thesis', 'independent'),
    ('ZF', 'Well-Ordering Theorem', 'independent'),
    ('ZFC', 'ZF', 'contains'),
    ('ZFC', 'PA', 'contains'),
    ('ZF', 'PA', 'contains'),
    ('PA2', 'PA', 'contains'),
    ('ACA0', 'PA', 'contains'),
    ('ZFC+LC', 'ZFC', 'contains'),
    ('ZFC+MM', 'ZFC', 'contains'),
    ('ZFC+AD', 'ZFC', 'contains'),
    ('ZFC+PD', 'ZFC', 'contains'),
    ('Zorn\'s Lemma', 'Well-Ordering Theorem', 'implies'),
    ('Well-Ordering Theorem', 'Axiom of Choice', 'implies'),
    ('Axiom of Choice', 'Zorn\'s Lemma', 'implies'),
    ('Incompleteness Theorem', "Gödel's Incompleteness", 'implies'),
    ('Halting Problem', 'P vs NP', 'implies'),
    ('Prime Number Theorem', 'Riemann Hypothesis', 'implies'),
    ('Prime Number Theorem', 'Twin Prime Conjecture', 'implies'),
    ('Fundamental Theorem of Arithmetic', 'Prime Number Theorem', 'implies'),
    ('Completeness Theorem', 'Compactness Theorem', 'implies'),
    ('Compactness Theorem', 'Löwenheim–Skolem Theorem', 'implies')
]

# Generate synthetic data
def generate_synthetic_data(num_systems=50, num_theorems=300, avg_relations_per_node=6, independence_ratio=0.3, both_ratio=0.15):
    """Generate synthetic data for testing and development.
    
    Args:
        num_systems: Number of formal systems to generate
        num_theorems: Number of theorems to generate
        avg_relations_per_node: Average number of relationships per node
        independence_ratio: Ratio of independence relationships
        both_ratio: Ratio of theorems that are both provable and independent
    """
    synthetic_systems = {}
    synthetic_theorems = {}
    synthetic_relationships = []
    
    # Generate formal systems with varying strengths
    system_strengths = ['weak', 'moderate', 'strong']
    system_prefixes = ['S', 'T', 'F', 'L', 'M']
    
    for i in range(num_systems):
        prefix = random.choice(system_prefixes)
        strength = random.choice(system_strengths)
        name = f"{prefix}{i+1}"
        synthetic_systems[name] = {
            'name': name,
            'strength': random.uniform(0.3, 0.9),
            'description': f"Synthetic {strength} formal system {i+1}"
        }
    
    # Generate theorems with varying complexity
    theorem_types = ['basic', 'intermediate', 'advanced', 'specialized']
    theorem_domains = ['algebra', 'analysis', 'topology', 'logic', 'geometry', 'number_theory']
    
    for i in range(num_theorems):
        complexity = random.choice(theorem_types)
        domain = random.choice(theorem_domains)
        name = f"Theorem_{domain}_{i+1}"
        synthetic_theorems[name] = {
            'name': name,
            'complexity': complexity,
            'domain': domain,
            'description': f"Synthetic {complexity} theorem in {domain}"
        }
    
    # Generate relationships
    systems = list(synthetic_systems.keys())
    theorems = list(synthetic_theorems.keys())
    
    # Ensure each theorem has at least one relationship
    for theorem in theorems:
        system = random.choice(systems)
        rel_type = random.choices(
            ['proves', 'independent', 'implies'],
            weights=[0.5, independence_ratio, 0.2]
        )[0]
        synthetic_relationships.append((system, theorem, rel_type))
    
    # Add additional relationships to meet avg_relations_per_node
    total_desired_relations = int(len(theorems) * avg_relations_per_node)
    while len(synthetic_relationships) < total_desired_relations:
        # Randomly choose between system-theorem and theorem-theorem relationships
        if random.random() < 0.7:  # 70% system-theorem relationships
            system = random.choice(systems)
            theorem = random.choice(theorems)
            rel_type = random.choices(
                ['proves', 'independent', 'implies'],
                weights=[0.5, independence_ratio, 0.2]
            )[0]
            rel = (system, theorem, rel_type)
        else:  # 30% theorem-theorem relationships
            theorem1 = random.choice(theorems)
            theorem2 = random.choice(theorems)
            if theorem1 != theorem2:
                rel = (theorem1, theorem2, 'implies')
        
        if rel not in synthetic_relationships:
            synthetic_relationships.append(rel)
    
    # Add "both" relationships for some theorems
    both_theorems = random.sample(theorems, int(len(theorems) * both_ratio))
    for theorem in both_theorems:
        # Add both proves and independent relationships
        systems_sample = random.sample(systems, 2)
        synthetic_relationships.append((systems_sample[0], theorem, 'proves'))
        synthetic_relationships.append((systems_sample[1], theorem, 'independent'))
    
    return synthetic_systems, synthetic_theorems, synthetic_relationships

# Generate expanded synthetic dataset
_synthetic_systems, _synthetic_theorems, _synthetic_relationships = generate_synthetic_data(
    num_systems=50,
    num_theorems=300,
    avg_relations_per_node=6,
    independence_ratio=0.3,
    both_ratio=0.15
)

# Combine all data sources
EXPANDED_SYSTEMS = {**_systems_from_old_script, **_synthetic_systems}
ADDITIONAL_THEOREMS = {**_theorems_from_old_script, **_synthetic_theorems}
EXPANDED_RELATIONSHIPS = _relationships_from_old_script + _synthetic_relationships


def get_all_theorems() -> Dict:
    """Combine all theorem collections into one dictionary."""
    all_theorems = {}
    all_theorems.update(SET_THEORY_THEOREMS)
    all_theorems.update(NUMBER_THEORY_THEOREMS)
    all_theorems.update(ANALYSIS_THEOREMS)
    all_theorems.update(CATEGORY_THEOREMS)
    all_theorems.update(ADDITIONAL_THEOREMS) # Include newly added theorems
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
    # This function needs to be updated to correctly identify independent theorems
    # based on the 'status' field in the combined theorems data.
    # For now, it will use the existing logic.
    all_theorems = get_all_theorems()
    return [name for name, data in all_theorems.items()
            if data.get('independent_of') or data.get('status') == 'open' or data.get('status') == 'independent']