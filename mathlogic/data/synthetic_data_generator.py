import networkx as nx
import random
from typing import Dict, List, Tuple, Any

def generate_synthetic_data(
    num_systems: int = 15,
    num_theorems: int = 50,
    avg_relations_per_node: int = 3,
    independence_ratio: float = 0.2, # % of theorems that are independent
    both_ratio: float = 0.1 # % of theorems that are 'both'
) -> Tuple[Dict, Dict, List[Tuple[str, str, str]]]:
    """
    Generates a synthetic dataset of mathematical systems, theorems, and relationships.
    
    Args:
        num_systems: Number of formal systems to generate.
        num_theorems: Number of theorems to generate.
        avg_relations_per_node: Average number of relationships per node.
        independence_ratio: Proportion of theorems to be marked as 'independent'.
        both_ratio: Proportion of theorems to be marked as 'both'.
        
    Returns:
        A tuple containing:
        - systems_data (Dict): Dictionary of synthetic formal systems.
        - theorems_data (Dict): Dictionary of synthetic theorems.
        - relationships_data (List): List of synthetic relationships.
    """
    systems_data = {}
    theorems_data = {}
    relationships_data = []

    # 1. Generate Systems
    for i in range(num_systems):
        sys_name = f"System{i+1}"
        systems_data[sys_name] = {
            'description': f"Synthetic formal system {i+1}",
            'strength': round(random.uniform(0.5, 1.5), 2)
        }

    # 2. Generate Theorems
    theorem_names = [f"Theorem{i+1}" for i in range(num_theorems)]
    
    # Classify theorems for better distribution
    independent_theorems_count = int(num_theorems * independence_ratio)
    both_theorems_count = int(num_theorems * both_ratio)
    provable_theorems_count = num_theorems - independent_theorems_count - both_theorems_count

    random.shuffle(theorem_names)
    
    independent_set = set(theorem_names[:independent_theorems_count])
    both_set = set(theorem_names[independent_theorems_count : independent_theorems_count + both_theorems_count])
    provable_set = set(theorem_names[independent_theorems_count + both_theorems_count:])

    for name in theorem_names:
        status = "unknown"
        if name in independent_set:
            status = "independent"
        elif name in both_set:
            status = "both"
        elif name in provable_set:
            status = "provable" # This will be refined by 'proves' relations

        theorems_data[name] = {
            'description': f"Synthetic theorem {name}",
            'status': status,
            'complexity': random.choice(['low', 'medium', 'high', 'very_high']),
            'field': random.choice(['Set Theory', 'Number Theory', 'Logic', 'Analysis', 'Topology'])
        }

    all_nodes = list(systems_data.keys()) + list(theorems_data.keys())
    
    # 3. Generate Relationships
    # Create a temporary graph to manage connections and avoid duplicates
    temp_G = nx.DiGraph()
    temp_G.add_nodes_from(all_nodes)

    possible_relations = ["proves", "contains", "implies", "independent", "contradicts"]

    for node in all_nodes:
        num_relations = random.randint(1, avg_relations_per_node * 2) # Vary number of relations
        
        for _ in range(num_relations):
            target = random.choice(all_nodes)
            if node == target: # Avoid self-loops for most relations
                continue
            
            relation_type = random.choice(possible_relations)

            # Ensure logical consistency for 'contains' and 'proves'
            if relation_type == "contains":
                if node in systems_data and target in systems_data:
                    if not temp_G.has_edge(node, target):
                        relationships_data.append((node, target, relation_type))
                        temp_G.add_edge(node, target)
            elif relation_type == "proves":
                if node in systems_data and target in theorems_data:
                    if not temp_G.has_edge(node, target):
                        relationships_data.append((node, target, relation_type))
                        temp_G.add_edge(node, target)
            elif relation_type == "implies":
                if node in theorems_data and target in theorems_data:
                    if not temp_G.has_edge(node, target):
                        relationships_data.append((node, target, relation_type))
                        temp_G.add_edge(node, target)
            elif relation_type == "independent":
                if not temp_G.has_edge(node, target) and not temp_G.has_edge(target, node):
                    relationships_data.append((node, target, relation_type))
                    temp_G.add_edge(node, target) # Add edge to prevent duplicate independent relations
            elif relation_type == "contradicts":
                if not temp_G.has_edge(node, target) and not temp_G.has_edge(target, node):
                    relationships_data.append((node, target, relation_type))
                    temp_G.add_edge(node, target)

    print(f"Generated {len(systems_data)} systems, {len(theorems_data)} theorems, and {len(relationships_data)} relationships.")
    return systems_data, theorems_data, relationships_data

if __name__ == "__main__":
    # Generate a larger dataset for testing
    systems, theorems, relationships = generate_synthetic_data(
        num_systems=20,
        num_theorems=100,
        avg_relations_per_node=5,
        independence_ratio=0.25,
        both_ratio=0.05
    )
    
    # Save to a JSON file for inspection
    output_data = {
        "systems": systems,
        "theorems": theorems,
        "relationships": relationships
    }
    
    output_filepath = os.path.join("entailment_output", "synthetic_data.json")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Synthetic data saved to {output_filepath}")