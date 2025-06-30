"""
Script to expand the entailment dataset with additional theorems and relationships.
"""

import json
import os
from typing import Dict, List, Set, Tuple
from datetime import datetime
from entailment_theory import EntailmentCone
from data_validation import validate_formal_system, validate_theorem, expand_entailment_cone
from theorem_expansion import ADDITIONAL_THEOREMS, ADDITIONAL_SYSTEMS, INDEPENDENCE_RELATIONSHIPS
from expanded_theorems import (
    get_all_theorems,
    get_all_systems,
    get_all_relationships,
    get_theorems_by_field,
    get_independent_theorems,
    SET_THEORY_THEOREMS,
    NUMBER_THEORY_THEOREMS,
    ANALYSIS_THEOREMS,
    CATEGORY_THEOREMS
)

def expand_dataset() -> Tuple[EntailmentCone, Dict]:
    """Expand the dataset with additional theorems and relationships."""
    # Create a new entailment cone
    cone = EntailmentCone("Extended Mathematical Logic")
    
    # First add base theorems and systems
    cone, base_report = expand_entailment_cone(cone)
    
    # Add theorems by field
    fields = ['set_theory', 'number_theory', 'analysis', 'category_theory']
    field_counts = {}
    
    for field in fields:
        field_theorems = get_theorems_by_field(field)
        for name, data in field_theorems.items():
            if name not in cone.statements and validate_theorem(name, data):
                cone.add_statement(name, data)
                field_counts[field] = field_counts.get(field, 0) + 1
    
    # Add expanded systems that don't already exist
    expanded_systems = get_all_systems()
    for name, data in expanded_systems.items():
        if name not in cone.formal_systems and validate_formal_system(name, data):
            cone.add_formal_system(name, data)
    
    # Add relationships
    relationships = get_all_relationships()
    for source, target, rel_type in relationships:
        if source in cone.statements and target in cone.statements:
            try:
                cone.add_relationship(source, target, rel_type)
            except ValueError:
                # Skip if relationship already exists
                pass
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_statements': len(cone.statements),
        'total_relationships': len(cone.relations),
        'field_distribution': field_counts,
        'independent_theorems': len(get_independent_theorems()),
        'formal_systems': len(cone.formal_systems)
    }
    
    # Save expanded dataset
    save_expanded_dataset(cone, report)
    
    return cone, report

def save_expanded_dataset(cone: EntailmentCone, report: Dict) -> None:
    """Save the expanded dataset and report."""
    output_dir = 'entailment_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save report
    report_path = os.path.join(output_dir, f'expansion_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save graph data
    graph_path = os.path.join(output_dir, 'expanded_entailment_graph.json')
    graph_data = {
        'statements': {k: v.to_dict() for k, v in cone.statements.items()},
        'relations': [rel.to_dict() for rel in cone.relations],
        'formal_systems': cone.formal_systems
    }
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)

if __name__ == '__main__':
    cone, report = expand_dataset()
    print(f"Expanded dataset created with {report['total_statements']} statements and {report['total_relationships']} relationships") 