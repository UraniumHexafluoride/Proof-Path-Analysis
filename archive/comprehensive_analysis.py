import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from collections import defaultdict
import matplotlib.cm as cm
from independence_results import create_independence_entailment_graph, detect_bottlenecks

# Use absolute path for output directory
OUTPUT_DIR = os.path.abspath("entailment_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def expand_entailment_graph(G):
    """
    Expand the entailment graph with additional theorems and formal systems.
    """
    # Add more formal systems
    additional_systems = {
        "ZFC+PD": "ZFC with Projective Determinacy",
        "ZFC+MM": "ZFC with Martin's Maximum",
        "ZFC+AD": "ZFC with Axiom of Determinacy",
        "ZF+DC": "ZF with Dependent Choice",
        "Z": "Zermelo set theory",
        "PA2": "Second-order Peano Arithmetic",
        "HA": "Heyting Arithmetic",
        "WKL0": "Weak König's Lemma",
    }
    
    for system, description in additional_systems.items():
        G.add_node(system, type="system", description=description)
    
    # Add more theorems and statements
    additional_theorems = {
        "Ramsey": "Ramsey's Theorem",
        "KP": "Kruskal-Penrose Theorem",
        "BT": "Borel Determinacy",
        "PD": "Projective Determinacy",
        "MC": "Measurable Cardinal exists",
        "Ind_Comp": "Independence of Comprehension Schema",
        "FLT": "Fermat's Last Theorem",
        "4CT": "Four Color Theorem",
        "Con(ZF)": "Consistency of ZF",
    }
    
    for theorem, description in additional_theorems.items():
        G.add_node(theorem, type="theorem", description=description)
    
    # Add more entailment relationships
    additional_entailments = [
        # System containment
        ("ZFC+PD", "ZFC", "Contains"),
        ("ZFC+MM", "ZFC", "Contains"),
        ("ZFC+AD", "ZFC", "Contains"),
        ("ZF+DC", "ZF", "Contains"),
        ("Z", "ZF", "Contained_in"),
        ("PA2", "PA", "Contains"),
        ("PA", "HA", "Contains"),
        ("ACA0", "WKL0", "Contains"),
        ("WKL0", "RCA0", "Contains"),
        
        # Provability relationships
        ("PA", "Ramsey", "Proves"),
        ("ACA0", "Ramsey", "Proves"),
        ("PA2", "KP", "Independence"),
        ("ZFC", "BT", "Proves"),
        ("ZFC", "PD", "Independence"),
        ("ZFC+PD", "PD", "Contains"),
        ("ZFC+LC", "PD", "Proves"),
        ("ZFC+LC", "MC", "Contains"),
        ("ZFC+MM", "PD", "Proves"),
        ("ZF", "Con(ZF)", "Independence"),
        ("PA2", "FLT", "Proves"),  # Simplified for illustration
        ("PA", "4CT", "Independence"),  # Simplified for illustration
        ("ACA0", "4CT", "Proves"),
    ]
    
    for source, target, relation in additional_entailments:
        G.add_edge(source, target, relation=relation)
    
    return G

def visualize_entailment_cone(G, output_file="entailment_cone.png"):
    """
    Create a visualization of the entailment cone.
    """
    plt.figure(figsize=(16, 12))
    
    # Create a layout that emphasizes the hierarchical structure
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Get node types for coloring
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('type') == 'system':
            node_colors.append('skyblue')
        else:
            node_colors.append('lightgreen')
    
    # Get edge types for coloring
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['relation'] == 'Independence':
            edge_colors.append('red')
        elif data['relation'] == 'Proves':
            edge_colors.append('green')
        elif data['relation'] == 'Contains':
            edge_colors.append('blue')
        elif data['relation'] == 'Contained_in':
            edge_colors.append('purple')
        elif data['relation'] == 'Disproves':
            edge_colors.append('orange')
        else:
            edge_colors.append('gray')
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], color='skyblue', marker='o', markersize=15, label='Formal System', linestyle='None'),
        plt.Line2D([0], [0], color='lightgreen', marker='o', markersize=15, label='Theorem', linestyle='None'),
        plt.Line2D([0], [0], color='red', label='Independence'),
        plt.Line2D([0], [0], color='green', label='Proves'),
        plt.Line2D([0], [0], color='blue', label='Contains'),
        plt.Line2D([0], [0], color='purple', label='Contained_in'),
        plt.Line2D([0], [0], color='orange', label='Disproves')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Mathematical Entailment Cone", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    filepath = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Entailment cone visualization saved to {filepath}")
    return filepath

def analyze_proof_paths(G):
    """
    Analyze proof paths between formal systems and theorems.
    """
    # Identify all theorems
    theorems = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'theorem']
    
    # Identify all formal systems
    systems = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'system']
    
    # Find proof paths from each system to each theorem
    proof_paths = {}
    for system in systems:
        for theorem in theorems:
            try:
                # Find all simple paths from system to theorem
                paths = list(nx.all_simple_paths(G, system, theorem, cutoff=5))
                if paths:
                    if system not in proof_paths:
                        proof_paths[system] = {}
                    proof_paths[system][theorem] = paths
            except nx.NetworkXNoPath:
                continue
    
    # Save proof paths to CSV
    filepath = os.path.join(OUTPUT_DIR, "proof_paths.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["System", "Theorem", "Path Length", "Path"])
        
        for system in proof_paths:
            for theorem in proof_paths[system]:
                for path in proof_paths[system][theorem]:
                    path_str = " -> ".join(path)
                    writer.writerow([system, theorem, len(path), path_str])
    
    print(f"Proof path analysis saved to {filepath}")
    
    # Find minimal axiom extensions needed for each theorem
    minimal_axioms = {}
    for theorem in theorems:
        # Find all systems that can prove this theorem
        proving_systems = []
        for system in systems:
            if system in proof_paths and theorem in proof_paths[system]:
                proving_systems.append(system)
        
        # Find minimal systems (those not contained in others)
        minimal_systems = []
        for system1 in proving_systems:
            is_minimal = True
            for system2 in proving_systems:
                if system1 != system2 and nx.has_path(G, system2, system1):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_systems.append(system1)
        
        minimal_axioms[theorem] = minimal_systems
    
    # Save minimal axioms to CSV
    filepath = os.path.join(OUTPUT_DIR, "minimal_axioms.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Theorem", "Minimal Axiom Systems"])
        
        for theorem, systems in minimal_axioms.items():
            writer.writerow([theorem, ", ".join(systems)])
    
    print(f"Minimal axiom analysis saved to {filepath}")
    
    return proof_paths, minimal_axioms

def perform_topological_analysis(G):
    """
    Perform basic topological analysis of the entailment cone.
    """
    # Create undirected graph for some analyses
    G_undirected = G.to_undirected()
    
    # Calculate basic topological features
    results = {
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "connected_components": nx.number_connected_components(G_undirected),
        "average_clustering": nx.average_clustering(G_undirected),
        "transitivity": nx.transitivity(G_undirected),
    }
    
    # Calculate cycle basis (fundamental cycles)
    try:
        cycle_basis = nx.cycle_basis(G_undirected)
        results["cycle_count"] = len(cycle_basis)
        results["average_cycle_length"] = np.mean([len(cycle) for cycle in cycle_basis]) if cycle_basis else 0
    except:
        results["cycle_count"] = "Error computing cycles"
        results["average_cycle_length"] = "Error computing cycles"
    
    # Calculate graph density
    results["density"] = nx.density(G)
    
    # Calculate diameter (if connected)
    if results["connected_components"] == 1:
        results["diameter"] = nx.diameter(G_undirected)
    else:
        results["diameter"] = "Graph is not connected"
    
    # Save results to CSV
    filepath = os.path.join(OUTPUT_DIR, "topological_analysis.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        
        for metric, value in results.items():
            writer.writerow([metric, value])
    
    print(f"Topological analysis saved to {filepath}")
    
    # Create a visualization highlighting connected components
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Color nodes by connected component
    components = list(nx.connected_components(G_undirected))
    colors = cm.rainbow(np.linspace(0, 1, len(components)))
    
    node_colors = []
    for node in G.nodes():
        for i, component in enumerate(components):
            if node in component:
                node_colors.append(colors[i])
                break
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title("Connected Components in Entailment Cone", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    filepath = os.path.join(OUTPUT_DIR, "connected_components.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Connected components visualization saved to {filepath}")
    
    return results

if __name__ == "__main__":
    print("Creating and expanding independence entailment graph...")
    G = create_independence_entailment_graph()
    G = expand_entailment_graph(G)
    
    print("\nVisualizing entailment cone...")
    visualize_entailment_cone(G)
    
    print("\nAnalyzing bottlenecks...")
    analysis = detect_bottlenecks(G)
    
    print("\nAnalyzing proof paths...")
    proof_paths, minimal_axioms = analyze_proof_paths(G)
    
    print("\nPerforming topological analysis...")
    topo_results = perform_topological_analysis(G)
    
    print("\nAnalysis complete! Results saved to the entailment_output directory.")
    print("\nSummary of findings:")
    print(f"- Graph contains {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"- Found {len(analysis['cut_vertices'])} cut vertices (critical points)")
    print(f"- Found {len(analysis['independence_bridges'])} independence bridges")
    print(f"- Identified minimal axiom systems for {len(minimal_axioms)} theorems")
    print(f"- Graph has {topo_results['connected_components']} connected components")
    print(f"- Graph contains approximately {topo_results.get('cycle_count', 'unknown')} fundamental cycles")