import networkx as nx
import csv
import os

# Get output directory from environment or use default
OUTPUT_DIR = os.environ.get('ENTAILMENT_OUTPUT_DIR', '.')

# Load graph from CSV
def load_graph_from_csv(filename="entailment_graph_updated.csv"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    G = nx.DiGraph()
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            source, target, relation = row
            G.add_edge(source, target, relation=relation)
    return G

# Enumerate proof paths with rule tracking
def enumerate_proof_paths(G, start, end, cutoff=5):
    paths = []
    try:
        # Use shortest_path for minimal derivations
        shortest_path = nx.shortest_path(G, source=start, target=end)
        path_with_rules = []
        for i in range(len(shortest_path) - 1):
            source, target = shortest_path[i], shortest_path[i + 1]
            relation = G.edges[source, target]['relation']
            path_with_rules.append(f"{source} -> {target} ({relation})")
        paths.append(path_with_rules)
        
        # Optionally add all simple paths (comment out if too large)
        # all_paths = nx.all_simple_paths(G, start, end, cutoff)
        # for path in all_paths:
        #     path_with_rules = []
        #     for i in range(len(path) - 1):
        #         source, target = path[i], path[i + 1]
        #         relation = G.edges[source, target]['relation']
        #         path_with_rules.append(f"{source} -> {target} ({relation})")
        #     paths.append(path_with_rules)
    except nx.NetworkXNoPath:
        return []
    return paths

# Save paths to CSV
def save_paths_to_csv(paths, start, end, filename="proof_paths.csv"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "End", "Proof Path"])
        for path in paths:
            writer.writerow([start, end, " | ".join(path)])
    print(f"Proof paths saved to {os.path.abspath(filepath)}")
    return filepath

# Main execution
if __name__ == "__main__":
    G = load_graph_from_csv()
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")

    start_node = "P"
    end_node = "R"
    paths = enumerate_proof_paths(G, start_node, end_node)
    
    if paths:
        print(f"Found {len(paths)} proof path(s) from {start_node} to {end_node}:")
        for p in paths:
            print(" | ".join(p))
    else:
        print(f"No proof paths found from {start_node} to {end_node}.")
    
    save_paths_to_csv(paths, start_node, end_node)
