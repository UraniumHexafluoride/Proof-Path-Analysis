"""
Theorem URL mappings and utilities for handling theorem names and URLs.
"""

from typing import Dict, Set
import re

# Known alternative URLs for common theorems
THEOREM_URL_MAPPINGS = {
    # Original mappings
    'Brouwer_Fixed_Point_Theorem': 'Brouwer%27s_Fixed_Point_Theorem',
    'Schroeder_Bernstein_Theorem': 'Schr%C3%B6der-Bernstein_Theorem',
    'Jordan_Holder_Theorem': 'Jordan-H%C3%B6lder_Theorem',
    'Riemann_Mapping_Theorem': 'Riemann%27s_Mapping_Theorem',
    'Cayley_Hamilton_Theorem': 'Cayley-Hamilton_Theorem',
    'Stone_Weierstrass_Theorem': 'Stone-Weierstrass_Theorem',
    'Arzela_Ascoli_Theorem': 'Arzel%C3%A0-Ascoli_Theorem',
    'Hahn_Banach_Theorem': 'Hahn-Banach_Theorem',
    'Banach_Steinhaus_Theorem': 'Banach-Steinhaus_Uniform_Boundedness_Theorem',
    'Krein_Milman_Theorem': 'Krein-Milman_Theorem',
    'Bolzano_Weierstrass_Theorem': 'Bolzano-Weierstrass_Theorem',
    
    # New mappings
    'Euler_Totient_Function': 'Euler%27s_Totient_Function',
    'Quadratic_Reciprocity': 'Law_of_Quadratic_Reciprocity',
    'Finitely_Generated_Module': 'Finitely_Generated_Module_over_Ring',
    'Least_Absolute_Remainder': 'Least_Absolute_Remainder_Theorem',
    'Abel_Generalisation': 'Abel%27s_Generalization_of_Binomial_Theorem',
    'Riesz_Representation_Theorem': 'Riesz_Representation_Theorem_for_Hilbert_Spaces',
    'Spectral_Theorem': 'Spectral_Theorem_for_Normal_Operators',
    'Rank_Nullity_Theorem': 'Rank-Nullity_Theorem',
    'Fundamental_Theorem_of_Linear_Algebra': 'Fundamental_Theorems_of_Linear_Algebra',
}

# Pages to skip (navigation, metadata, etc.)
SKIP_PAGES = {
    'Lemma',
    'Corollary',
    'Proof',
    'Historical_Note',
    'General',
    'Variant_Form',
    'Also_known_as',
    'General_Theorem',
}

def should_skip_page(page_name: str) -> bool:
    """
    Check if a page should be skipped (navigation, metadata, etc.)
    """
    # Check exact matches
    if page_name in SKIP_PAGES:
        return True
        
    # Check patterns
    skip_patterns = [
        r'^Lemma_\d+$',
        r'^Corollary_\d+$',
        r'^Proof_\d+$',
    ]
    
    return any(re.match(pattern, page_name) for pattern in skip_patterns)

def normalize_theorem_url(theorem_name: str) -> str:
    """
    Normalize a theorem name to its ProofWiki URL format.
    """
    # Check direct mappings first
    if theorem_name in THEOREM_URL_MAPPINGS:
        return THEOREM_URL_MAPPINGS[theorem_name]
        
    # Apply general normalization rules
    normalized = theorem_name.replace('_', ' ')
    normalized = re.sub(r'\s+', '_', normalized.strip())
    
    return normalized 