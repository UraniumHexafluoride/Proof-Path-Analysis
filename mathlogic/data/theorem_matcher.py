"""
Theorem matching and deduplication system for multi-source mathematical data collection.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from difflib import SequenceMatcher
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TheoremIdentity:
    canonical_name: str
    alternate_names: Set[str]
    source_urls: Set[str]
    description: str
    metadata: Dict[str, any]
    
class TheoremMatcher:
    def __init__(self):
        self.known_theorems: Dict[str, TheoremIdentity] = {}
        self.name_variations: Dict[str, str] = {}  # Maps variations to canonical names
        
    def generate_canonical_name(self, name: str) -> str:
        """Generate a canonical name for a theorem."""
        # Remove special characters and standardize spacing
        canonical = re.sub(r'[^\w\s-]', '', name)
        canonical = re.sub(r'\s+', ' ', canonical)
        canonical = canonical.lower().strip()
        
        # Handle common variations
        canonical = re.sub(r"'s\b|s'\b|'", "", canonical)  # Handle possessives
        canonical = re.sub(r"theorem$", "", canonical)  # Remove "theorem" suffix
        canonical = canonical.strip()
        
        # Handle known equivalent names
        equivalents = {
            "fundamental theorem of calculus": "ftc",
            "fundamental theorem of algebra": "fta",
            "central limit theorem": "clt",
            "pythagorean theorem": "pythagoras",
            # Add more equivalents as needed
        }
        
        return equivalents.get(canonical, canonical)
    
    def are_theorems_similar(self, name1: str, name2: str, threshold: float = 0.85) -> bool:
        """Check if two theorem names are similar using fuzzy matching."""
        # Get canonical forms
        can1 = self.generate_canonical_name(name1)
        can2 = self.generate_canonical_name(name2)
        
        # Direct match on canonical names
        if can1 == can2:
            return True
            
        # Sequence matcher for fuzzy matching
        similarity = SequenceMatcher(None, can1, can2).ratio()
        return similarity >= threshold
    
    def add_theorem(self, name: str, source_url: str, description: str, metadata: Dict[str, any]) -> str:
        """
        Add a new theorem to the database or update existing one.
        Returns the canonical name used for the theorem.
        """
        canonical_name = self.generate_canonical_name(name)
        
        # Check for existing similar theorems
        for existing_canonical, identity in self.known_theorems.items():
            if self.are_theorems_similar(canonical_name, existing_canonical):
                # Update existing theorem
                identity.alternate_names.add(name)
                identity.source_urls.add(source_url)
                # Merge descriptions if new one is more detailed
                if len(description) > len(identity.description):
                    identity.description = description
                # Update metadata
                identity.metadata.update(metadata)
                return existing_canonical
        
        # No match found, create new theorem identity
        self.known_theorems[canonical_name] = TheoremIdentity(
            canonical_name=canonical_name,
            alternate_names={name},
            source_urls={source_url},
            description=description,
            metadata=metadata
        )
        
        # Add to name variations mapping
        self.name_variations[name.lower()] = canonical_name
        
        return canonical_name
    
    def get_theorem_identity(self, name: str) -> Optional[TheoremIdentity]:
        """Get theorem identity by any known name variation."""
        canonical = self.generate_canonical_name(name)
        return self.known_theorems.get(canonical)
    
    def merge_theorems(self, name1: str, name2: str) -> str:
        """
        Manually merge two theorems that are determined to be the same.
        Returns the canonical name of the merged theorem.
        """
        id1 = self.get_theorem_identity(name1)
        id2 = self.get_theorem_identity(name2)
        
        if not id1 or not id2:
            raise ValueError("One or both theorems not found")
            
        # Use the more common name as canonical
        if len(id1.source_urls) >= len(id2.source_urls):
            primary, secondary = id1, id2
        else:
            primary, secondary = id2, id1
            
        # Merge identities
        primary.alternate_names.update(secondary.alternate_names)
        primary.source_urls.update(secondary.source_urls)
        if len(secondary.description) > len(primary.description):
            primary.description = secondary.description
        primary.metadata.update(secondary.metadata)
        
        # Update mappings
        del self.known_theorems[secondary.canonical_name]
        for alt_name in secondary.alternate_names:
            self.name_variations[alt_name.lower()] = primary.canonical_name
            
        return primary.canonical_name
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the theorem database."""
        return {
            "total_theorems": len(self.known_theorems),
            "total_variations": len(self.name_variations),
            "multi_source_theorems": sum(1 for t in self.known_theorems.values() if len(t.source_urls) > 1),
            "avg_variations_per_theorem": sum(len(t.alternate_names) for t in self.known_theorems.values()) / len(self.known_theorems) if self.known_theorems else 0
        } 