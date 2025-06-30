"""
Integration module for the expanded theorem dataset.
This module combines the original and expanded datasets, updates relationships,
and provides enhanced analysis capabilities.
"""

import os
import networkx as nx
from typing import Dict, List, Tuple, Set
from entailment_theory import EntailmentCone
from data_validation import validate_formal_system, validate_theorem
from expanded_theorems import (
    get_all_theorems,
    get_all_systems,
    get_all_relationships
)
from expanded_theorems_v2 import (
    get_all_theorems_v2,
    get_all_systems_v2,
    get_all_relationships_v2,
    ALGEBRAIC_GEOMETRY_THEOREMS,
    DIFFERENTIAL_GEOMETRY_THEOREMS,
    MODEL_THEORY_THEOREMS,
    PROOF_THEORY_THEOREMS,
    ALGEBRAIC_TOPOLOGY_THEOREMS,
    HOMOLOGICAL_ALGEBRA_THEOREMS,
    DESCRIPTIVE_SET_THEORY_THEOREMS,
    RECURSION_THEORY_THEOREMS,
    EXPANDED_SYSTEMS_V2,
    EXPANDED_RELATIONSHIPS_V2
)

# Additional statements needed for relationships
ADDITIONAL_STATEMENTS = {
    'CH': {
        'description': 'Continuum Hypothesis',
        'field': 'set_theory',
        'complexity': 'high'
    },
    'AC': {
        'description': 'Axiom of Choice',
        'field': 'set_theory',
        'complexity': 'medium'
    },
    'PFA': {
        'description': 'Proper Forcing Axiom',
        'field': 'set_theory',
        'complexity': 'high'
    },
    'RCA0': {
        'description': 'Recursive Comprehension Axiom',
        'field': 'proof_theory',
        'complexity': 'medium'
    },
    'HOL': {
        'description': 'Higher Order Logic',
        'field': 'logic',
        'complexity': 'medium'
    },
    'IZF': {
        'description': 'Intuitionistic Zermelo-Fraenkel set theory',
        'field': 'set_theory',
        'complexity': 'high'
    },
    'Con(PA)': {
        'description': 'Consistency statement for Peano Arithmetic',
        'field': 'proof_theory',
        'complexity': 'high'
    },
    'Poincaré Conjecture': {
        'description': 'Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere',
        'field': 'topology',
        'complexity': 'high'
    },
    'Projective Determinacy': {
        'description': 'Every projective set is determined',
        'field': 'set_theory',
        'complexity': 'high'
    },
    'ZFC': {
        'description': 'Zermelo-Fraenkel set theory with Choice',
        'field': 'set_theory',
        'complexity': 'medium'
    },
    'PA': {
        'description': 'Peano Arithmetic',
        'field': 'number_theory',
        'complexity': 'medium'
    },
    'ZFC+I2': {
        'description': 'ZFC with second-order large cardinal axiom',
        'field': 'set_theory',
        'complexity': 'high'
    }
}

def create_integrated_entailment_cone() -> EntailmentCone:
    """Create an integrated entailment cone combining both datasets."""
    cone = EntailmentCone("Integrated Mathematical Knowledge Graph")
    
    # First add all statements needed for relationships
    for name, data in ADDITIONAL_STATEMENTS.items():
        if validate_theorem(name, data):
            try:
                cone.add_statement(name, data)
            except ValueError as e:
                print(f"Warning: Could not add additional statement {name}: {e}")
    
    # Add all theorems from both datasets
    theorems_v1 = get_all_theorems()
    theorems_v2 = get_all_theorems_v2()
    
    for name, data in theorems_v1.items():
        if validate_theorem(name, data):
            try:
                cone.add_statement(name, data)
            except ValueError as e:
                print(f"Warning: Could not add theorem {name} from v1: {e}")
    
    for name, data in theorems_v2.items():
        if validate_theorem(name, data):
            try:
                cone.add_statement(name, data)
            except ValueError as e:
                print(f"Warning: Could not add theorem {name} from v2: {e}")
    
    # Add all formal systems from both datasets
    systems_v1 = get_all_systems()
    systems_v2 = get_all_systems_v2()
    
    for name, data in systems_v1.items():
        if validate_formal_system(name, data):
            try:
                cone.add_formal_system(name, data)
            except ValueError as e:
                print(f"Warning: Could not add system {name} from v1: {e}")
    
    for name, data in systems_v2.items():
        if validate_formal_system(name, data):
            try:
                cone.add_formal_system(name, data)
            except ValueError as e:
                print(f"Warning: Could not add system {name} from v2: {e}")
    
    # Add all relationships from both datasets
    relationships_v1 = get_all_relationships()
    relationships_v2 = get_all_relationships_v2()
    
    for source, target, rel_type in relationships_v1:
        try:
            if source in cone.statements and target in cone.statements:
                cone.add_relationship(source, target, rel_type)
        except ValueError as e:
            print(f"Warning: Could not add relationship {source}->{target} from v1: {e}")
    
    for source, target, rel_type in relationships_v2:
        try:
            if source in cone.statements and target in cone.statements:
                cone.add_relationship(source, target, rel_type)
        except ValueError as e:
            print(f"Warning: Could not add relationship {source}->{target} from v2: {e}")
    
    return cone

def analyze_field_coverage(cone: EntailmentCone) -> Dict[str, Dict]:
    """Analyze the coverage of different mathematical fields."""
    fields = {
        'set_theory': 'Set Theory',
        'number_theory': 'Number Theory',
        'analysis': 'Analysis',
        'category_theory': 'Category Theory',
        'algebraic_geometry': 'Algebraic Geometry',
        'differential_geometry': 'Differential Geometry',
        'model_theory': 'Model Theory',
        'proof_theory': 'Proof Theory',
        'algebraic_topology': 'Algebraic Topology',
        'homological_algebra': 'Homological Algebra',
        'descriptive_set_theory': 'Descriptive Set Theory',
        'recursion_theory': 'Recursion Theory',
        'topology': 'Topology',
        'logic': 'Logic'
    }
    
    coverage = {}
    for field_id, field_name in fields.items():
        field_theorems = [stmt for stmt in cone.statements.values() 
                         if stmt.metadata.get('field') == field_id]
        
        if field_theorems:
            coverage[field_id] = {
                'name': field_name,
                'theorem_count': len(field_theorems),
                'proven_count': sum(1 for t in field_theorems if t.metadata.get('proven_by')),
                'independent_count': sum(1 for t in field_theorems if t.metadata.get('independent_of')),
                'open_count': sum(1 for t in field_theorems if t.metadata.get('status') == 'open'),
                'average_complexity': sum(
                    {'low': 1, 'medium': 2, 'high': 3}.get(t.metadata.get('complexity', 'medium'), 2)
                    for t in field_theorems
                ) / len(field_theorems)
            }
    
    return coverage

def analyze_system_hierarchy(cone: EntailmentCone) -> Dict[str, Dict]:
    """Analyze the hierarchy of formal systems."""
    systems = {}
    for name, system in cone.formal_systems.items():
        # Find systems that this system contains (direct relationships only)
        contained_systems = [
            target for target in cone.formal_systems
            if any(rel.source == name and rel.target == target and rel.relation_type == 'contains'
                  for rel in cone.relations)
        ]
        
        systems[name] = {
            'description': system.get('description', ''),
            'strength': system.get('strength', 0.0),
            'contains': contained_systems,
            'theorem_count': sum(1 for stmt in cone.statements.values()
                               if name in stmt.metadata.get('proven_by', [])),
            'independence_count': sum(1 for stmt in cone.statements.values()
                                   if name in stmt.metadata.get('independent_of', []))
        }
    
    return systems

def generate_integration_report(cone: EntailmentCone) -> str:
    """Generate a comprehensive report about the integrated dataset."""
    report = ["# Integrated Mathematical Knowledge Graph Analysis\n"]
    
    # Basic statistics
    report.append("## Dataset Statistics\n")
    report.append(f"- Total theorems: {len(cone.statements)}")
    report.append(f"- Total formal systems: {len(cone.formal_systems)}")
    report.append(f"- Total relationships: {len(cone.relations)}\n")
    
    # Field coverage analysis
    coverage = analyze_field_coverage(cone)
    report.append("## Field Coverage\n")
    report.append("| Field | Theorems | Proven | Independent | Open | Avg Complexity |")
    report.append("|-------|----------|--------|-------------|------|----------------|")
    for field_data in coverage.values():
        report.append(
            f"| {field_data['name']} | {field_data['theorem_count']} | "
            f"{field_data['proven_count']} | {field_data['independent_count']} | "
            f"{field_data['open_count']} | {field_data['average_complexity']:.2f} |"
        )
    report.append("")
    
    # System hierarchy analysis
    systems = analyze_system_hierarchy(cone)
    report.append("## Formal System Hierarchy\n")
    for name, data in systems.items():
        report.append(f"### {name}")
        report.append(f"- Description: {data['description']}")
        report.append(f"- Logical strength: {data['strength']:.2f}")
        report.append(f"- Contains: {', '.join(data['contains'])}")
        report.append(f"- Proves {data['theorem_count']} theorems")
        report.append(f"- Has {data['independence_count']} independence results\n")
    
    # Save report
    output_dir = "entailment_output"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "integration_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report_path

def main():
    """Main function to create and analyze the integrated dataset."""
    print("Creating integrated entailment cone...")
    cone = create_integrated_entailment_cone()
    
    print("\nValidating integrated structure...")
    validation = cone.validate_structure()
    if not validation['is_valid']:
        print("Warning: Validation found issues:")
        for msg in validation.get('messages', []):
            print(f"- {msg}")
    
    print("\nGenerating integration report...")
    report_path = generate_integration_report(cone)
    
    print(f"\nAnalysis complete! Results saved to: {report_path}")
    return cone

if __name__ == "__main__":
    main() 