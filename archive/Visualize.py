import networkx as nx
import csv

# Load entailment graph from CSV
def load_graph_from_csv(filename):
    G = nx.DiGraph()
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            source, target, relation = row
            G.add_edge(source, target, relation=relation)
    return G

# Identify cut nodes (removing them disconnects the graph)
def find_cut_nodes(G):
    undirected_G = G.to_undirected()  # Convert to undirected for articulation point detection
    return set(nx.articulation_points(undirected_G))  # Find cut nodes

# Find axioms required for proofs
def find_required_axioms(G, proof_paths):
    required_axioms = set()
    for path in proof_paths:
        for node in path[1:-1]:  # Ignore start & end nodes
            required_axioms.add(node)
    return required_axioms

# Load proof paths from CSV
def load_proof_paths(filename="bottlenecks.csv"):  # Updated to match proof_file
    proof_paths = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            proof_paths.append(row[2].split(" -> "))  # Extract proof path
    return proof_paths

# Save axiom dependencies to CSV
def save_axiom_dependencies(cut_nodes, required_axioms, filename="axiom_dependencies.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Axiom", "Is Cut Node?", "Required for Proofs"])
        all_axioms = cut_nodes.union(required_axioms)
        for axiom in all_axioms:
            writer.writerow([axiom, axiom in cut_nodes, axiom in required_axioms])

# Main execution
if __name__ == "__main__":
    input_file = "entailment_graph.csv"
    proof_file = "bottlenecks.csv"

    print("Loading graph...")
    G = load_graph_from_csv(input_file)
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    print("Finding cut nodes (critical axioms)...")
    cut_nodes = find_cut_nodes(G)
    print(f"Cut nodes detected: {cut_nodes}")
    
    print("Loading proof paths...")
    proof_paths = load_proof_paths(proof_file)
    
    print("Detecting required axioms...")
    required_axioms = find_required_axioms(G, proof_paths)
    print(f"Required axioms for proofs: {required_axioms}")
    
    save_axiom_dependencies(cut_nodes, required_axioms)
    print(f"Axiom dependencies saved to axiom_dependencies.csv.")
