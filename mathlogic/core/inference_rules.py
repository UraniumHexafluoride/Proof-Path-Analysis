import networkx as nx
import csv
import time
import networkx as nx
import time
import os
import csv # Keep csv for save_graph_to_csv

# Get output directory from environment or use default
OUTPUT_DIR = os.environ.get('ENTAILMENT_OUTPUT_DIR', 'entailment_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_modus_ponens(G):
    """
    Apply modus ponens (A and A → B implies B) to the graph.
    
    Args:
        G (nx.DiGraph): The entailment graph.
    
    Returns:
        int: Number of new edges added.
    """
    start_time = time.time()
    new_edges = []
    for node in list(G.nodes()): # Iterate over a copy to allow modification
        for neighbor in list(G.successors(node)): # Iterate over a copy
            # Check if the node itself is an implication (e.g., "A -> B")
            # This is a simplistic check and assumes a specific string format.
            # A more robust solution would involve a proper logical parser.
            if "→" in node:
                try:
                    antecedent, consequent = node.split("→", 1) # Split only on the first arrow
                    antecedent = antecedent.strip()
                    consequent = consequent.strip()

                    # If the neighbor is the antecedent and the consequent is not already a successor
                    if antecedent == neighbor and not G.has_edge(node, consequent):
                        new_edges.append((node, consequent, {"relation": "Modus Ponens"}))
                except ValueError:
                    # Handle cases where "→" is present but split fails (e.g., malformed string)
                    continue
    
    G.add_edges_from(new_edges)
    print(f"Modus Ponens applied: {len(new_edges)} new edges. Time: {time.time() - start_time:.4f}s")
    return len(new_edges)

def apply_reflexivity(G):
    """
    Apply reflexivity (A → A) to the graph.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        
    Returns:
        int: Number of new edges added.
    """
    start_time = time.time()
    new_edges = [(node, node, {"relation": "Reflexivity"}) for node in G.nodes() if not G.has_edge(node, node)]
    G.add_edges_from(new_edges)
    print(f"Reflexivity applied: {len(new_edges)} new edges. Time: {time.time() - start_time:.4f}s")
    return len(new_edges)

def save_graph_to_csv(G, filename="entailment_graph_inferred.csv"):
    """
    Save the updated graph to a CSV file.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        filename (str): Name of the output CSV file.
        
    Returns:
        str: Absolute path to the saved file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Entailment Relation"])
        writer.writerows((source, target, G.edges[source, target]['relation']) for source, target in G.edges())
    print(f"Graph saved to {os.path.abspath(filepath)}")
    return filepath

# Main execution block for demonstration or direct testing
if __name__ == "__main__":
    print("This script is intended to be imported and used as part of the mathlogic package.")
    print("For demonstration, it will attempt to create a dummy graph and apply rules.")
    
    # Create a dummy graph for demonstration purposes
    demo_G = nx.DiGraph()
    demo_G.add_nodes_from(["A", "B", "C", "A → B", "B → C"])
    demo_G.add_edge("A", "A → B", relation="Implies")
    demo_G.add_edge("A → B", "B", relation="Implies") # This edge makes Modus Ponens applicable
    demo_G.add_edge("B", "B → C", relation="Implies")
    demo_G.add_edge("B → C", "C", relation="Implies")

    print(f"Demo graph initialized: {len(demo_G.nodes())} nodes, {len(demo_G.edges())} edges.")

    # Apply logical inference rules
    new_mp_edges = apply_modus_ponens(demo_G)
    new_ref_edges = apply_reflexivity(demo_G)

    print(f"Total new edges added: {new_mp_edges + new_ref_edges}")
    
    # Save updated graph
    save_graph_to_csv(demo_G, "demo_entailment_graph_inferred.csv")
    print("Demo complete. Check 'entailment_output/demo_entailment_graph_inferred.csv'")
