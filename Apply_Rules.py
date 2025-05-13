import networkx as nx
import csv
import time
import os

# Get output directory from environment or use default
OUTPUT_DIR = os.environ.get('ENTAILMENT_OUTPUT_DIR', '.')

# Load entailment graph from CSV
def load_graph_from_csv(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    G = nx.DiGraph()
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            source, target, relation = row
            G.add_edge(source, target, relation=relation)
    return G

# Apply modus ponens (A and A → B implies B)
def apply_modus_ponens(G):
    start_time = time.time()
    new_edges = []
    for node in G.nodes():
        for neighbor in G.successors(node):
            if "→" in node:  # If node is an implication
                antecedent, consequent = node.split("→")
                if antecedent.strip() == neighbor and consequent.strip() not in G.successors(node):
                    new_edges.append((node, consequent.strip(), {"relation": "Modus Ponens"}))
    G.add_edges_from(new_edges)
    print(f"Modus Ponens applied: {len(new_edges)} new edges. Time: {time.time() - start_time:.4f}s")

# Apply reflexivity (A → A)
def apply_reflexivity(G):
    start_time = time.time()
    new_edges = [(node, node, {"relation": "Reflexivity"}) for node in G.nodes() if not G.has_edge(node, node)]
    G.add_edges_from(new_edges)
    print(f"Reflexivity applied: {len(new_edges)} new edges. Time: {time.time() - start_time:.4f}s")

# Save updated graph to CSV
def save_graph_to_csv(G, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Entailment Relation"])
        writer.writerows((source, target, G.edges[source, target]['relation']) for source, target in G.edges())
    print(f"Graph saved to {os.path.abspath(filepath)}")
    return filepath

# Main execution
if __name__ == "__main__":
    input_file = "entailment_graph.csv"
    output_file = "entailment_graph_updated.csv"

    print("Loading graph...")
    start_time = time.time()
    G = load_graph_from_csv(input_file)
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges. Time: {time.time() - start_time:.4f}s")

    # Apply logical inference rules
    apply_modus_ponens(G)
    apply_reflexivity(G)

    # Save updated graph
    save_graph_to_csv(G, output_file)
    print(f"Updated graph saved as {output_file}")
