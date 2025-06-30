import networkx as nx
import csv
import os

# Output directory for reports and figures
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_cut_nodes(G):
    """
    Identify cut nodes (articulation points) in the graph.
    Removing a cut node increases the number of connected components.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        
    Returns:
        set: A set of cut nodes.
    """
    undirected_G = G.to_undirected()  # Convert to undirected for articulation point detection
    return set(nx.articulation_points(undirected_G))  # Find cut nodes

def find_required_axioms(G, proof_paths):
    """
    Find axioms required for proofs based on given proof paths.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        proof_paths (list): A list of proof paths, where each path is a list of nodes.
        
    Returns:
        set: A set of axioms required for the proofs.
    """
    required_axioms = set()
    for path in proof_paths:
        # Assuming axioms are typically at the beginning or intermediate steps
        # For simplicity, considering all intermediate nodes as potential required axioms
        for node in path[1:-1]:  # Ignore start & end nodes, focus on intermediate
            # A more robust check would involve checking node type if available
            required_axioms.add(node)
    return required_axioms

def save_axiom_dependencies(cut_nodes, required_axioms, filename="axiom_dependencies.csv"):
    """
    Save axiom dependencies to a CSV file.
    
    Args:
        cut_nodes (set): Set of cut nodes.
        required_axioms (set): Set of required axioms.
        filename (str): Name of the output CSV file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Axiom", "Is Cut Node?", "Required for Proofs"])
        all_axioms = cut_nodes.union(required_axioms)
        for axiom in all_axioms:
            writer.writerow([axiom, axiom in cut_nodes, axiom in required_axioms])
    print(f"Axiom dependencies saved to {os.path.abspath(filepath)}.")

# Main execution block (for demonstration or direct testing)
if __name__ == "__main__":
    # This block is for standalone testing. In a full application,
    # the graph and proof paths would likely be passed from a main orchestrator.
    print("This script is intended to be called with a pre-loaded graph and proof paths.")
    print("For demonstration, it will attempt to load from default CSVs.")
    
    # Placeholder for graph loading (replace with actual graph creation/loading in main app)
    try:
        from mathlogic.analysis.structural import create_expanded_entailment_graph
        G = create_expanded_entailment_graph()
        print(f"Demo graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    except ImportError:
        print("Could not import create_expanded_entailment_graph. Please ensure mathlogic package is set up.")
        print("Attempting to load from 'entailment_graph.csv' (if available)...")
        # Fallback for testing if structural.py is not yet integrated
        try:
            def load_graph_from_csv_temp(filename):
                temp_G = nx.DiGraph()
                temp_filepath = os.path.join(OUTPUT_DIR, filename)
                if not os.path.exists(temp_filepath):
                    print(f"Error: {temp_filepath} not found. Cannot load graph.")
                    return None
                with open(temp_filepath, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    for row in reader:
                        source, target, relation = row
                        temp_G.add_edge(source, target, relation=relation)
                return temp_G
            G = load_graph_from_csv_temp("entailment_graph.csv")
            if G:
                print(f"Graph loaded from CSV: {len(G.nodes())} nodes, {len(G.edges())} edges")
            else:
                exit("Graph loading failed. Exiting.")
        except Exception as e:
            exit(f"Error loading graph from CSV: {e}")

    # Placeholder for proof paths loading (replace with actual proof path generation)
    proof_paths_demo = [
        ["ZFC", "Zorn's Lemma", "Well-Ordering Theorem", "Axiom of Choice"],
        ["PA", "Fundamental Theorem of Arithmetic", "Prime Number Theorem"]
    ]
    print(f"Using demo proof paths: {len(proof_paths_demo)} paths")
    
    print("Finding cut nodes (critical axioms)...")
    cut_nodes = find_cut_nodes(G)
    print(f"Cut nodes detected: {cut_nodes}")
    
    print("Detecting required axioms...")
    required_axioms = find_required_axioms(G, proof_paths_demo)
    print(f"Required axioms for proofs: {required_axioms}")
    
    save_axiom_dependencies(cut_nodes, required_axioms)