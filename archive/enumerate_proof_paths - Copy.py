import networkx as nx
import csv

# Generate a two-way graph (from previous response)
def generate_two_way_graph():
    G = nx.DiGraph()
    # Two-way rules as edges in both directions
    G.add_edge("P", "Q", relation="P <-> Q")
    G.add_edge("Q", "P", relation="P <-> Q")
    G.add_edge("Q", "R", relation="Q <-> R")
    G.add_edge("R", "Q", relation="Q <-> R")
    # Reflexivity for completeness
    for node in ["P", "Q", "R"]:
        G.add_edge(node, node, relation="Reflexivity")
    return G

# Save graph to CSV (optional, for reference)
def save_graph_to_csv(G, filename="two_way_graph.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Entailment Relation"])
        for source, target, data in G.edges(data=True):
            writer.writerow([source, target, data["relation"]])

# Enumerate all proof paths with rule tracking
def enumerate_proof_paths(G, start, end, cutoff=5):
    paths = []
    try:
        for path in nx.all_simple_paths(G, source=start, target=end, cutoff=cutoff):
            path_with_rules = []
            for i in range(len(path) - 1):
                source, target = path[i], path[i + 1]
                relation = G.edges[source, target]["relation"]
                path_with_rules.append(f"{source} -> {target} ({relation})")
            paths.append(path_with_rules)
    except nx.NetworkXNoPath:
        return []
    return paths

# Save paths to CSV
def save_paths_to_csv(paths, start, end, filename="proof_paths_two_way.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "End", "Proof Path"])
        for path in paths:
            writer.writerow([start, end, " | ".join(path)])

# Main execution
if __name__ == "__main__":
    # Generate the two-way graph
    G = generate_two_way_graph()
    save_graph_to_csv(G)  # Save for reference
    print(f"Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Define start and end nodes
    start_node = "P"
    end_node = "R"
    
    # Enumerate paths
    paths = enumerate_proof_paths(G, start_node, end_node, cutoff=5)
    
    # Print and save results
    if paths:
        print(f"Found {len(paths)} proof path(s) from {start_node} to {end_node}:")
        for p in paths:
            print(" | ".join(p))
    else:
        print(f"No proof paths found from {start_node} to {end_node}.")
    
    save_paths_to_csv(paths, start_node, end_node)