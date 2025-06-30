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

# Identify bottlenecks using betweenness centrality
def detect_bottlenecks(G, top_n=5):
    betweenness = nx.betweenness_centrality(G)
    sorted_bottlenecks = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {node for node, _ in sorted_bottlenecks}  # Set of bottleneck nodes

# Find proof paths that pass through bottlenecks
def find_bottleneck_paths(G, bottlenecks):
    paths_data = []
    for start in G.nodes():
        for end in G.nodes():
            if start != end:
                try:
                    paths = list(nx.all_simple_paths(G, source=start, target=end, cutoff=5))  # Limit to 5 steps
                    for path in paths:
                        bottlenecks_in_path = [node for node in path[1:-1] if node in bottlenecks]
                        if bottlenecks_in_path:  # Only include paths with bottlenecks
                            paths_data.append([start, end, " -> ".join(path), ", ".join(bottlenecks_in_path)])
                except nx.NetworkXNoPath:
                    continue
    return paths_data

# Save bottleneck paths to CSV
def save_bottleneck_paths(paths_data, filename="bottlenecks.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "End", "Proof Path", "Bottlenecks in Path"])
        writer.writerows(paths_data)

# Main execution
if __name__ == "__main__":
    input_file = "entailment_graph.csv"
    output_file = "bottlenecks.csv"
    
    print("Loading graph...")
    G = load_graph_from_csv(input_file)
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    print("Detecting bottlenecks...")
    bottlenecks = detect_bottlenecks(G)
    print(f"Identified bottlenecks: {bottlenecks}")
    
    print("Finding proof paths through bottlenecks...")
    bottleneck_paths = find_bottleneck_paths(G, bottlenecks)
    print(f"Found {len(bottleneck_paths)} bottleneck paths")
    
    save_bottleneck_paths(bottleneck_paths, output_file)
    print(f"Bottleneck paths saved to {output_file}")

