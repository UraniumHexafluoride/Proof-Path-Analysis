"""
Data validation and expansion utilities for the entailment graph.
"""

import json
import os
from typing import Dict, List, Set, Tuple
from datetime import datetime
from entailment_theory import EntailmentCone, LogicalStatement, EntailmentRelation
from theorem_expansion import (
    ADDITIONAL_THEOREMS as KNOWN_THEOREMS,
    ADDITIONAL_SYSTEMS as FORMAL_SYSTEMS,
    INDEPENDENCE_RELATIONSHIPS
)

# Known formal systems and their relationships
FORMAL_SYSTEMS = {
    'ZFC': {
        'description': 'Zermelo-Fraenkel set theory with Choice',
        'contains': ['ZF'],
        'strength': 1.0
    },
    'ZF': {
        'description': 'Zermelo-Fraenkel set theory',
        'contains': ['Z'],
        'strength': 0.9
    },
    'PA': {
        'description': 'Peano Arithmetic',
        'contains': ['HA'],
        'strength': 0.8
    },
    'PA2': {
        'description': 'Second-order Peano Arithmetic',
        'contains': ['PA'],
        'strength': 0.85
    },
    'ZFC+LC': {
        'description': 'ZFC with Large Cardinals',
        'contains': ['ZFC'],
        'strength': 1.1
    },
    'ZFC+MM': {
        'description': "ZFC with Martin's Maximum",
        'contains': ['ZFC'],
        'strength': 1.15
    }
}

# Known theorems and their properties
KNOWN_THEOREMS = {
    "Fermat's Last Theorem": {
        'description': 'No three positive integers a, b, and c satisfy the equation aⁿ + bⁿ = cⁿ for any integer n > 2',
        'proven_by': ['PA2'],
        'year_proven': 1995,
        'complexity': 'high'
    },
    'Four Color Theorem': {
        'description': 'Any planar map can be colored with four colors such that no adjacent regions share the same color',
        'proven_by': ['PA'],
        'year_proven': 1976,
        'complexity': 'medium'
    },
    'Continuum Hypothesis': {
        'description': 'There is no set whose cardinality is strictly between that of the integers and the real numbers',
        'independent_of': ['ZFC'],
        'proven_independent': 1963,
        'complexity': 'high'
    },
    'AC': {
        'description': 'For any collection of non-empty sets, there exists a function that selects one element from each set',
        'proven_by': ['ZFC'],
        'independent_of': ['ZF'],
        'year_proven': 1938,
        'complexity': 'medium',
        'also_known_as': ['Axiom of Choice']
    }
}

def validate_formal_system(name: str, data: Dict) -> bool:
    """
    Validate a formal system and its metadata.
    
    Args:
        name: Name of the formal system
        data: Dictionary containing system metadata
        
    Returns:
        True if valid, False otherwise
    """
    # Check name
    if not isinstance(name, str) or not name:
        return False
        
    # Check required fields
    if 'description' not in data:
        return False
        
    # Check contains field if present
    if 'contains' in data and not isinstance(data['contains'], list):
        return False
        
    # Check strength if present
    if 'strength' in data:
        try:
            strength = float(data['strength'])
            if strength < 0 or strength > 2:  # Reasonable range for system strength
                return False
        except (ValueError, TypeError):
            return False
            
    return True

def validate_theorem(name: str, data: Dict) -> bool:
    """
    Validate a theorem and its metadata.
    
    Args:
        name: Name of the theorem
        data: Dictionary containing theorem metadata
        
    Returns:
        True if valid, False otherwise
    """
    # Check name
    if not isinstance(name, str) or not name:
        return False
        
    # Check required fields
    if 'description' not in data:
        return False
        
    # Check field type if present
    if 'field' in data and not isinstance(data['field'], str):
        return False
        
    # Check complexity if present
    if 'complexity' in data and data['complexity'] not in ['low', 'medium', 'high']:
        return False
        
    # Check year if present
    if 'year_proven' in data:
        try:
            year = int(data['year_proven'])
            if year < 1600 or year > 2100:  # Reasonable range for mathematical proofs
                return False
        except (ValueError, TypeError):
            return False
            
    return True

def expand_entailment_cone(cone: EntailmentCone) -> Tuple[EntailmentCone, Dict]:
    """
    Expand an entailment cone with base theorems and systems.
    
    Args:
        cone: The entailment cone to expand
        
    Returns:
        Tuple of (expanded cone, expansion report)
    """
    report = {
        'added_statements': [],
        'added_relations': [],
        'validation_messages': []
    }
    
    # Add base theorems
    for name, data in KNOWN_THEOREMS.items():
        try:
            if validate_theorem(name, data):
                cone.add_statement(name, data)
                report['added_statements'].append(name)
        except Exception as e:
            report['validation_messages'].append(f"Error adding theorem {name}: {str(e)}")
    
    # Add formal systems
    for name, data in FORMAL_SYSTEMS.items():
        try:
            if validate_formal_system(name, data):
                cone.add_formal_system(name, data)
                report['added_statements'].append(name)
        except Exception as e:
            report['validation_messages'].append(f"Error adding system {name}: {str(e)}")
    
    # Add independence relationships
    for source, target, rel_type in INDEPENDENCE_RELATIONSHIPS:
        try:
            if source in cone.statements and target in cone.statements:
                cone.add_relationship(source, target, rel_type)
                report['added_relations'].append(f"{source} -{rel_type}-> {target}")
        except Exception as e:
            report['validation_messages'].append(f"Error adding relationship {source} -> {target}: {str(e)}")
    
    return cone, report

def save_validation_report(report: Dict, output_dir: str = "entailment_output"):
    """Save a validation report to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return filepath

if __name__ == "__main__":
    # Create a new entailment cone
    cone = EntailmentCone("Mathematical Knowledge Graph")
    
    # Expand it with validated data
    expanded_cone, report = expand_entailment_cone(cone)
    
    # Save validation report
    report_path = save_validation_report(report)
    
    print(f"Validation report saved to: {report_path}")
    print(f"Added {len(report['added_statements'])} statements")
    print(f"Added {len(report['added_relations'])} relations")
    print(f"Validation messages: {len(report['validation_messages'])}")
    
    # Validate final structure
    validation = expanded_cone.validate_structure()
    print("\nFinal validation results:")
    print(f"Valid: {validation['is_valid']}")
    print(f"Errors: {len(validation['errors'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    print(f"Total statements: {validation['metrics']['statement_count']}")
    print(f"Total relations: {validation['metrics']['relation_count']}") 