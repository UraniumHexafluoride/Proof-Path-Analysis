import csv
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
from sympy import Symbol, And, Or, Not, Implies

def generate_basic_entailment_csv(filename="entailment_graph.csv"):
    # Basic symbols
    symbols = ['A', 'B', 'C', 'D', 'E', 'F']
    variables = [Symbol(s) for s in symbols]
    
    # Define axioms using SymPy
    axioms = [
        (variables[0], variables[1], 'Entailment'),  # A → B
        (variables[1], variables[2], 'Entailment'),  # B → C
        (And(variables[0], variables[1]), variables[2], 'Entailment'),  # (A ∧ B) → C
        (variables[2], variables[3], 'Entailment'),  # C → D
        (variables[3], variables[4], 'Entailment'),  # D → E
        (variables[4], variables[5], 'Entailment'),  # E → F
        (variables[5], variables[0], 'Entailment'),  # F → A
        (Not(variables[1]), Not(variables[0]), 'Contraposition'),  # ¬B → ¬A
    ]
    
    # Add disjunction (OR) formulas
    for i in range(len(variables) - 1):
        axioms.append((
            Or(variables[i], variables[i+1]),
            variables[i+1],
            'Disjunction Elimination'
        ))
    
    # Add implication formulas
    for i in range(len(variables) - 2):
        axioms.append((
            Implies(variables[i], variables[i+1]),
            Implies(variables[i+1], variables[i+2]),
            'Transitivity'
        ))

    # Write to CSV — convert formulas to string labels
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Entailment Relation"])
        for source, target, relation in axioms:
            writer.writerow([str(source), str(target), relation])

    print(f"Generated {len(axioms)} structured axioms and wrote to {filename}.")

def generate_foundational_entailment_graph(filename="foundational_entailment.csv"):
    """
    Generate a foundational entailment graph with mathematical systems and independence results.
    """
    # Define mathematical systems and statements
    systems = {
        "ZFC": "Zermelo-Fraenkel with Choice",
        "PA": "Peano Arithmetic",
        "RCA0": "Recursive Comprehension Axiom",
        "WKL0": "Weak König's Lemma",
        "ACA0": "Arithmetical Comprehension Axiom",
        "ATR0": "Arithmetical Transfinite Recursion",
        "PH": "Paris-Harrington Theorem",
        "GS": "Goodstein's Theorem",
        "CH": "Continuum Hypothesis",
        "AC": "Axiom of Choice",
        "LCH": "Large Cardinal Hypothesis",
        "CON_ZFC": "Consistency of ZFC"
    }
    
    # Define entailment relationships
    entailments = [
        # Reverse mathematics hierarchy
        ("RCA0", "WKL0", "Subsystem"),
        ("WKL0", "ACA0", "Subsystem"),
        ("ACA0", "ATR0", "Subsystem"),
        
        # Independence results
        ("PA", "PH", "Independence"),  # Paris-Harrington is independent of PA
        ("PA", "GS", "Independence"),  # Goodstein's Theorem is independent of PA
        ("ZFC", "CH", "Independence"),  # Continuum Hypothesis is independent of ZFC
        
        # Provability relationships
        ("ZFC", "PA", "Proves"),
        ("ACA0", "PH", "Proves"),      # ACA0 proves Paris-Harrington
        ("ACA0", "GS", "Proves"),      # ACA0 proves Goodstein's Theorem
        ("LCH", "CON_ZFC", "Proves"),  # Large cardinals prove consistency of ZFC
        
        # Logical implications
        ("CH", "AC", "Implies"),       # CH implies certain forms of AC
        ("ZFC", "AC", "Contains"),     # ZFC contains AC
    ]
    
    # Write to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Relation Type", "Description"])
        for source, target, relation in entailments:
            source_desc = systems.get(source, "")
            target_desc = systems.get(target, "")
            writer.writerow([source, target, relation, f"{source_desc} → {target_desc}"])
    
    print(f"Generated foundational entailment graph with {len(systems)} systems and {len(entailments)} relationships")
    return systems, entailments

def visualize_entailment_cone(systems, entailments, title="Mathematical Entailment Cone"):
    """
    Visualize the entailment cone as a directed graph.
    """
    G = nx.DiGraph()
    
    # Add nodes with system descriptions
    for system, description in systems.items():
        G.add_node(system, description=description)
    
    # Add edges with relationship types
    for source, target, relation in entailments:
        G.add_edge(source, target, relation=relation)
    
    # Set up node colors based on system type
    node_colors = []
    for node in G.nodes():
        if node in ["ZFC", "PA"]:
            node_colors.append("lightblue")  # Base theories
        elif node in ["RCA0", "WKL0", "ACA0", "ATR0"]:
            node_colors.append("lightgreen")  # Reverse math systems
        elif node in ["PH", "GS", "CH"]:
            node_colors.append("salmon")  # Independence results
        elif node in ["LCH"]:
            node_colors.append("purple")  # Strong axioms
        else:
            node_colors.append("lightgray")
    
    # Set up edge colors based on relation type
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['relation'] == "Independence":
            edge_colors.append("red")
        elif data['relation'] == "Proves":
            edge_colors.append("green")
        elif data['relation'] == "Subsystem":
            edge_colors.append("blue")
        else:
            edge_colors.append("gray")
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # Consistent layout
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add edge labels
    edge_labels = {(u, v): data['relation'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("entailment_cone.png", dpi=300)
    plt.close()  # Close the figure to avoid display in non-interactive environments
    
    return G

def analyze_bottlenecks(G):
    """
    Identify bottlenecks in the entailment cone.
    """
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Find cut vertices (articulation points)
    undirected_G = G.to_undirected()
    cut_vertices = list(nx.articulation_points(undirected_G))
    
    # Identify nodes that bridge independence results
    independence_bridges = []
    for u, v, data in G.edges(data=True):
        if data['relation'] == "Independence":
            # Find nodes that can prove the target
            for node in G.nodes():
                if node != u and node != v:
                    if nx.has_path(G, node, v) and not nx.has_path(G, u, v):
                        independence_bridges.append((node, v))
    
    return {
        "betweenness": betweenness,
        "cut_vertices": cut_vertices,
        "independence_bridges": independence_bridges
    }

if __name__ == "__main__":
    # Generate basic entailment CSV using SymPy
    generate_basic_entailment_csv()
    
    # Generate and visualize the entailment cone
    systems, entailments = generate_foundational_entailment_graph()
    G = visualize_entailment_cone(systems, entailments)
    
    # Analyze the structure
    analysis = analyze_bottlenecks(G)
    
    print("\nBottleneck Analysis:")
    print(f"Cut vertices: {analysis['cut_vertices']}")
    print("\nBetweenness Centrality (Top 5):")
    sorted_betweenness = sorted(analysis['betweenness'].items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in sorted_betweenness:
        print(f"  {node}: {score:.4f}")
    
    print("\nIndependence Bridges:")
    for source, target in analysis['independence_bridges']:
        print(f"  {source} bridges to {target}")