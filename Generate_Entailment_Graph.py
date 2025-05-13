import networkx as nx
import csv
import os
import traceback

# Use absolute path for output directory
OUTPUT_DIR = os.path.abspath("entailment_output")
print(f"Using output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate a basic entailment graph CSV if it doesn't exist
def generate_basic_entailment_csv():
    filepath = os.path.join(OUTPUT_DIR, "entailment_graph.csv")
    print(f"Generating basic entailment graph at: {filepath}")
    
    try:
        # Create a simple graph with basic logical relationships
        G = nx.DiGraph()
        
        # Add some basic logical statements and relationships
        G.add_edge("P", "Q", relation="Modus Ponens")
        G.add_edge("P", "P → Q", relation="Cosubstitution")
        G.add_edge("Q", "R", relation="Modus Ponens")
        G.add_edge("P → Q", "Q", relation="Modus Ponens")
        G.add_edge("Q → R", "R", relation="Modus Ponens")
        
        # Save the graph
        with open(filepath, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Source", "Target", "Entailment Relation"])
            for source, target, data in G.edges(data=True):
                writer.writerow([source, target, data["relation"]])
        
        print(f"Basic entailment graph generated at {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        return G
    except Exception as e:
        print(f"ERROR generating basic entailment graph: {type(e).__name__}: {e}")
        traceback.print_exc()
        return nx.DiGraph()  # Return empty graph on error

# Load entailment graph from CSV
def load_graph_from_csv(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    G = nx.DiGraph()
    try:
        with open(filepath, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                source, target, relation = row
                G.add_edge(source, target, relation=relation)
        print(f"Successfully loaded graph from {filepath}")
    except FileNotFoundError:
        print(f"File not found: {filepath}, creating new graph")
        if filename == "entailment_graph.csv":
            G = generate_basic_entailment_csv()
    except Exception as e:
        print(f"Error loading graph: {e}, creating new graph")
    return G


# Apply substitution rule
def apply_substitution_single(node, G):
    new_edges = []
    for other in G.nodes():
        if node != other and node in other and len(node) < len(other):  
            new_edges.append((node, other, {'relation': 'Substitution'}))
    return new_edges

# Apply cosubstitution rule
def apply_cosubstitution_single(node, G):
    new_edges = []
    for other in G.nodes():
        if node != other and other in node and len(other) < len(node):  
            new_edges.append((other, node, {'relation': 'Cosubstitution'}))
    return new_edges

# Apply transitivity rule (only apply if necessary)
def apply_transitivity_single(node, G):
    new_edges = []
    for neighbor in G.successors(node):
        for second_neighbor in G.successors(neighbor):
            if not G.has_edge(node, second_neighbor):  # Prevent redundancy
                new_edges.append((node, second_neighbor, {'relation': 'Transitivity'}))
    return new_edges

# Apply reflexivity rule (Only for fundamental nodes)
def apply_reflexivity_single(node, G):
    new_edges = []
    if not G.has_edge(node, node):  # Prevent duplicate reflexivity edges
        new_edges.append((node, node, {'relation': 'Reflexivity'}))
    return new_edges

### OPTIMIZED MULTIPROCESSING ###
def process_node_for_rule(node, rule_function, G):
    return rule_function(node, G)

def apply_rule_parallel(rule_function, G):
    new_edges = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_node_for_rule, G.nodes(), [rule_function] * len(G.nodes()), [G] * len(G.nodes()))
        for result in results:
            new_edges.extend(result)
    return new_edges

# Save updated graph to CSV
def save_graph_to_csv(G, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"Attempting to save graph to: {filepath}")
    try:
        with open(filepath, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Source", "Target", "Entailment Relation"])
            for source, target in G.edges():
                writer.writerow([source, target, G.edges[source, target]['relation']])
        print(f"Graph successfully saved to {filepath}")
        print(f"File exists after save: {os.path.exists(filepath)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        return filepath
    except Exception as e:
        print(f"ERROR saving graph to {filepath}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

### MAIN EXECUTION ###
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    
    # Generate the basic entailment graph
    G = generate_basic_entailment_csv()
    
    # List all files in the output directory
    print("\nFiles in output directory:")
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        if files:
            for filename in files:
                file_path = os.path.join(OUTPUT_DIR, filename)
                file_size = os.path.getsize(file_path)
                print(f"  - {filename} ({file_size} bytes)")
        else:
            print("  No files found in the output directory.")
    else:
        print("  Output directory does not exist.")
