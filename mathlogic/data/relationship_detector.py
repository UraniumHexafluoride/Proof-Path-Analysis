"""
Detects relationships between mathematical theorems based on various indicators.
"""

from typing import Dict, List, Set, Tuple
import re
from dataclasses import dataclass
from mathlogic.data.enhanced_relationship_detector import EnhancedRelationshipDetector

@dataclass
class TheoremRelationship:
    source: str
    target: str
    relationship_type: str  # 'proves', 'implies', 'independent', 'related'
    confidence: float
    evidence: List[str]

class RelationshipDetector:
    def __init__(self):
        # Keywords indicating relationships
        self.proof_indicators = {
            'proves': ['proves', 'proof of', 'demonstrated by', 'shown by'],
            'implies': ['implies', 'leads to', 'follows from', 'consequence of'],
            'independent': ['independent of', 'not provable from', 'unprovable in'],
            'related': ['related to', 'connected with', 'analogous to']
        }
        
        # Common axiom systems for independence analysis
        self.axiom_systems = {
            'ZFC': ['zfc', 'zermelo-fraenkel', 'choice axiom'],
            'PA': ['peano arithmetic', 'peano axioms'],
            'ZF': ['zermelo-fraenkel']
        }
    
    def detect_relationships(self, theorem_data: Dict[str, Dict]) -> List[TheoremRelationship]:
        """Detect relationships between theorems based on descriptions and metadata."""
        relationships = []
        
        for name1, data1 in theorem_data.items():
            description1 = data1['description'].lower()
            
            # Look for explicit relationships in description
            for name2, data2 in theorem_data.items():
                if name1 == name2:
                    continue
                
                # Check for explicit mentions
                relationships.extend(self._check_explicit_relationships(
                    name1, name2, description1, data1, data2
                ))
                
                # Check for common axiom systems
                relationships.extend(self._check_axiom_relationships(
                    name1, name2, data1, data2
                ))
                
                # Check for field relationships
                relationships.extend(self._check_field_relationships(
                    name1, name2, data1, data2
                ))
        
        return relationships
    
    def _check_explicit_relationships(
        self, name1: str, name2: str, desc1: str, 
        data1: Dict, data2: Dict
    ) -> List[TheoremRelationship]:
        """Check for explicit mentions of relationships in descriptions."""
        relationships = []
        name2_variants = [name2.lower()] + [alt.lower() for alt in data2['alternate_names']]
        
        # Check if theorem2 is mentioned in theorem1's description
        for variant in name2_variants:
            if variant in desc1:
                # Check for relationship type based on surrounding words
                for rel_type, indicators in self.proof_indicators.items():
                    for indicator in indicators:
                        pattern = f"{indicator}.*{variant}|{variant}.*{indicator}"
                        if re.search(pattern, desc1):
                            relationships.append(TheoremRelationship(
                                source=name1,
                                target=name2,
                                relationship_type=rel_type,
                                confidence=0.8,
                                evidence=[f"Found '{indicator}' near '{variant}' in description"]
                            ))
        
        return relationships
    
    def _check_axiom_relationships(
        self, name1: str, name2: str, 
        data1: Dict, data2: Dict
    ) -> List[TheoremRelationship]:
        """Check for relationships based on common axiom systems."""
        relationships = []
        
        # Extract axiom systems mentioned in metadata
        axioms1 = set()
        axioms2 = set()
        
        for system, keywords in self.axiom_systems.items():
            desc1 = data1['description'].lower()
            desc2 = data2['description'].lower()
            
            for keyword in keywords:
                if keyword in desc1:
                    axioms1.add(system)
                if keyword in desc2:
                    axioms2.add(system)
        
        # If theorems share axiom systems, they might be related
        common_axioms = axioms1.intersection(axioms2)
        if common_axioms:
            relationships.append(TheoremRelationship(
                source=name1,
                target=name2,
                relationship_type='related',
                confidence=0.6,
                evidence=[f"Share axiom systems: {', '.join(common_axioms)}"]
            ))
        
        return relationships
    
    def _check_field_relationships(
        self, name1: str, name2: str, 
        data1: Dict, data2: Dict
    ) -> List[TheoremRelationship]:
        """Check for relationships based on mathematical fields."""
        relationships = []
        
        # Extract fields from metadata
        fields1 = set(cat.lower() for cat in data1.get('metadata', {}).get('categories', []))
        fields2 = set(cat.lower() for cat in data2.get('metadata', {}).get('categories', []))
        
        # If theorems share fields, they might be related
        common_fields = fields1.intersection(fields2)
        if common_fields:
            relationships.append(TheoremRelationship(
                source=name1,
                target=name2,
                relationship_type='related',
                confidence=0.4,
                evidence=[f"Share mathematical fields: {', '.join(common_fields)}"]
            ))
        
        return relationships

    def save_relationships(self, relationships: List[TheoremRelationship], output_file: str):
        """Save detected relationships to a JSON file."""
        import json
        
        relationship_data = [
            {
                'source': r.source,
                'target': r.target,
                'type': r.relationship_type,
                'confidence': r.confidence,
                'evidence': r.evidence
            }
            for r in relationships
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relationship_data, f, indent=2, ensure_ascii=False) 