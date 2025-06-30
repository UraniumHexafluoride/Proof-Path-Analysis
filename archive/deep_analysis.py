import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.cm as cm

# Use absolute path for output directory
OUTPUT_DIR = os.path.abspath("entailment_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_graph_from_csv(filename="independence_graph.csv"):
    """Load the entailment graph from CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    G = nx.DiGraph()
    
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            source, target, relation = row
            G.add_edge(source, target, relation=relation)
    
    # Add node types (system or theorem)
    systems = set()
    for source, target, relation in G.edges(data='relation'):
        if relation == 'Contains' or relation == 'Contained_in':
            systems.add(source)
            systems.add(target)
    
    for node in G.nodes():
        if node in systems:
            G.nodes[node]['type'] = 'system'
        else:
            G.nodes[node]['type'] = 'theorem'
    
    return G

def analyze_cut_vertices(G):
    """Identify and analyze cut vertices in the graph."""
    # Convert to undirected for cut vertex detection
    G_undirected = G.to_undirected()
    
    # Find cut vertices
    cut_vertices = list(nx.articulation_points(G_undirected))
    
    print(f"Found {len(cut_vertices)} cut vertices:")
    for vertex in cut_vertices:
        print(f"  - {vertex}")
    
    # Analyze the role of each cut vertex
    cut_vertex_analysis = {}
    for vertex in cut_vertices:
        # Remove the vertex and find connected components
        G_temp = G_undirected.copy()
        G_temp.remove_node(vertex)
        components = list(nx.connected_components(G_temp))
        
        # Analyze what gets disconnected
        systems_disconnected = []
        theorems_disconnected = []
        
        for component in components:
            sys_count = sum(1 for node in component if G.nodes[node].get('type') == 'system')
            thm_count = sum(1 for node in component if G.nodes[node].get('type') == 'theorem')
            
            if sys_count > 0:
                systems_disconnected.append(sys_count)
            if thm_count > 0:
                theorems_disconnected.append(thm_count)
        
        # Store analysis
        cut_vertex_analysis[vertex] = {
            'type': G.nodes[vertex].get('type', 'unknown'),
            'components_created': len(components),
            'systems_disconnected': systems_disconnected,
            'theorems_disconnected': theorems_disconnected
        }
    
    # Save analysis to CSV
    filepath = os.path.join(OUTPUT_DIR, "cut_vertex_analysis.csv")
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Cut Vertex", "Type", "Components Created", 
                         "Systems Disconnected", "Theorems Disconnected"])
        
        for vertex, data in cut_vertex_analysis.items():
            writer.writerow([
                vertex, 
                data['type'],
                data['components_created'],
                sum(data['systems_disconnected']),
                sum(data['theorems_disconnected'])
            ])
    
    print(f"Cut vertex analysis saved to {filepath}")
    
    # Visualize the cut vertices
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw regular nodes
    regular_nodes = [node for node in G.nodes() if node not in cut_vertices]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=regular_nodes,
                          node_color='lightblue', 
                          node_size=500, 
                          alpha=0.8)
    
    # Draw cut vertices with different color and size
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=cut_vertices,
                          node_color='red', 
                          node_size=800, 
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Cut Vertices in Entailment Graph", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    filepath = os.path.join(OUTPUT_DIR, "cut_vertices.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cut vertices visualization saved to {filepath}")
    
    return cut_vertices, cut_vertex_analysis

def analyze_components(G):
    """Analyze the connected components in the graph."""
    # Convert to undirected for component analysis
    G_undirected = G.to_undirected()
    
    # Find connected components
    components = list(nx.connected_components(G_undirected))
    
    print(f"Found {len(components)} connected components:")
    
    # Analyze each component
    component_analysis = []
    for i, component in enumerate(components):
        # Count systems and theorems
        systems = [node for node in component if G.nodes[node].get('type') == 'system']
        theorems = [node for node in component if G.nodes[node].get('type') == 'theorem']
        
        # Analyze the component
        analysis = {
            'component_id': i + 1,
            'size': len(component),
            'systems': len(systems),
            'theorems': len(theorems),
            'system_names': ', '.join(systems),
            'theorem_names': ', '.join(theorems),
            'density': nx.density(G.subgraph(component))
        }
        
        component_analysis.append(analysis)
        
        print(f"  Component {i+1}: {len(component)} nodes ({len(systems)} systems, {len(theorems)} theorems)")
    
    # Save analysis to CSV
    filepath = os.path.join(OUTPUT_DIR, "component_analysis.csv")
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Component ID", "Size", "Systems Count", "Theorems Count", 
                         "System Names", "Theorem Names", "Density"])
        
        for analysis in component_analysis:
            writer.writerow([
                analysis['component_id'],
                analysis['size'],
                analysis['systems'],
                analysis['theorems'],
                analysis['system_names'],
                analysis['theorem_names'],
                analysis['density']
            ])
    
    print(f"Component analysis saved to {filepath}")
    
    # Visualize the components
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Color nodes by component
    colors = cm.rainbow(np.linspace(0, 1, len(components)))
    
    for i, component in enumerate(components):
        # Draw nodes for this component
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(component),
                              node_color=[colors[i]] * len(component), 
                              node_size=500, 
                              alpha=0.8,
                              label=f"Component {i+1}")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Connected Components in Entailment Graph", fontsize=16)
    plt.legend()
    plt.axis('off')
    
    # Save the figure
    filepath = os.path.join(OUTPUT_DIR, "components_detailed.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Components visualization saved to {filepath}")
    
    return components, component_analysis

def analyze_minimal_axioms():
    """Analyze patterns in minimal axiom systems."""
    # Load minimal axioms from CSV
    filepath = os.path.join(OUTPUT_DIR, "minimal_axioms.csv")
    
    theorems = []
    axiom_systems = []
    
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            theorem, systems_str = row
            systems = [s.strip() for s in systems_str.split(',')]
            
            theorems.append(theorem)
            axiom_systems.append(systems)
    
    print(f"Loaded minimal axiom data for {len(theorems)} theorems")
    
    # Count frequency of each axiom system
    system_counts = Counter()
    for systems in axiom_systems:
        for system in systems:
            if system:  # Skip empty strings
                system_counts[system] += 1
    
    print("Most common minimal axiom systems:")
    for system, count in system_counts.most_common(5):
        print(f"  {system}: {count} theorems")
    
    # Group theorems by their minimal axiom systems
    theorem_groups = defaultdict(list)
    for theorem, systems in zip(theorems, axiom_systems):
        key = ', '.join(sorted(systems))
        theorem_groups[key].append(theorem)
    
    print(f"Found {len(theorem_groups)} distinct axiom requirement patterns")
    
    # Save grouping analysis to CSV
    filepath = os.path.join(OUTPUT_DIR, "theorem_groups.csv")
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Minimal Axiom Systems", "Theorems", "Count"])
        
        for systems, thms in theorem_groups.items():
            writer.writerow([systems, ', '.join(thms), len(thms)])
    
    print(f"Theorem grouping analysis saved to {filepath}")
    
    # Create a heatmap of theorems vs axiom systems
    # First, get unique systems across all theorems
    all_systems = set()
    for systems in axiom_systems:
        all_systems.update(systems)
    all_systems = sorted(list(all_systems))
    
    # Create a matrix of theorems vs systems
    matrix = np.zeros((len(theorems), len(all_systems)))
    for i, systems in enumerate(axiom_systems):
        for system in systems:
            if system in all_systems:
                j = all_systems.index(system)
                matrix[i, j] = 1
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    plt.imshow(matrix, cmap='Blues', aspect='auto')
    
    # Add labels
    plt.yticks(range(len(theorems)), theorems, fontsize=10)
    plt.xticks(range(len(all_systems)), all_systems, rotation=45, ha='right', fontsize=10)
    
    plt.title("Theorems vs Minimal Axiom Systems", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(OUTPUT_DIR, "theorem_axiom_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Theorem-axiom heatmap saved to {filepath}")
    
    return theorem_groups, system_counts

def analyze_cycles(G):
    """Analyze fundamental cycles in the graph."""
    # Convert to undirected for cycle analysis
    G_undirected = G.to_undirected()
    
    # Find cycle basis
    try:
        cycle_basis = nx.cycle_basis(G_undirected)
        print(f"Found {len(cycle_basis)} fundamental cycles")
        
        # Analyze each cycle
        cycle_analysis = []
        for i, cycle in enumerate(cycle_basis):
            # Count systems and theorems in the cycle
            systems = [node for node in cycle if G.nodes[node].get('type') == 'system']
            theorems = [node for node in cycle if G.nodes[node].get('type') == 'theorem']
            
            # Analyze the cycle
            analysis = {
                'cycle_id': i + 1,
                'length': len(cycle),
                'systems': len(systems),
                'theorems': len(theorems),
                'nodes': ', '.join(cycle)
            }
            
            cycle_analysis.append(analysis)
        
        # Save analysis to CSV
        filepath = os.path.join(OUTPUT_DIR, "cycle_analysis.csv")
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Cycle ID", "Length", "Systems Count", "Theorems Count", "Nodes"])
            
            for analysis in cycle_analysis:
                writer.writerow([
                    analysis['cycle_id'],
                    analysis['length'],
                    analysis['systems'],
                    analysis['theorems'],
                    analysis['nodes']
                ])
        
        print(f"Cycle analysis saved to {filepath}")
        
        # Visualize a few interesting cycles
        # Sort cycles by length (shorter cycles are often more interesting)
        sorted_cycles = sorted(enumerate(cycle_basis), key=lambda x: len(x[1]))
        
        # Visualize the 3 shortest cycles
        for i, (cycle_idx, cycle) in enumerate(sorted_cycles[:3]):
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42, k=0.5)
            
            # Draw regular nodes
            regular_nodes = [node for node in G.nodes() if node not in cycle]
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=regular_nodes,
                                  node_color='lightgray', 
                                  node_size=300, 
                                  alpha=0.5)
            
            # Draw cycle nodes
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=cycle,
                                  node_color='lightgreen', 
                                  node_size=600, 
                                  alpha=0.8)
            
            # Draw regular edges
            regular_edges = [(u, v) for u, v in G.edges() if u not in cycle or v not in cycle]
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=regular_edges,
                                  width=1.0, 
                                  alpha=0.2, 
                                  arrows=True)
            
            # Draw cycle edges
            cycle_edges = []
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                if G.has_edge(u, v):
                    cycle_edges.append((u, v))
                elif G.has_edge(v, u):
                    cycle_edges.append((v, u))
            
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=cycle_edges,
                                  width=2.0, 
                                  alpha=1.0, 
                                  arrows=True,
                                  edge_color='green')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title(f"Fundamental Cycle {cycle_idx+1} (Length {len(cycle)})", fontsize=16)
            plt.axis('off')
            
            # Save the figure
            filepath = os.path.join(OUTPUT_DIR, f"cycle_{cycle_idx+1}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Cycle {cycle_idx+1} visualization saved to {filepath}")
        
        return cycle_basis, cycle_analysis
    
    except Exception as e:
        print(f"Error analyzing cycles: {e}")
        return [], []

def generate_summary_report(cut_vertices, components, theorem_groups, system_counts, cycle_analysis):
    """Generate a comprehensive summary report of all analyses."""
    report = [
        "# Entailment Cone Analysis Report",
        "",
        "## 1. Cut Vertices Analysis",
        "",
        f"Found {len(cut_vertices)} cut vertices in the entailment graph. These represent critical points",
        "in the logical structure of mathematics, where removing them would disconnect parts of the graph.",
        "",
        "Top cut vertices:",
    ]
    
    for vertex in cut_vertices[:5]:
        report.append(f"- {vertex}")
    
    report.extend([
        "",
        "## 2. Connected Components Analysis",
        "",
        f"The entailment graph contains {len(components)} connected components, suggesting",
        "distinct areas of mathematical reasoning that are not directly connected.",
        "",
        "Component details:",
    ])
    
    for i, component in enumerate(components):
        systems = [node for node in component if G.nodes[node].get('type') == 'system']
        theorems = [node for node in component if G.nodes[node].get('type') == 'theorem']
        report.append(f"- Component {i+1}: {len(component)} nodes ({len(systems)} systems, {len(theorems)} theorems)")
    
    report.extend([
        "",
        "## 3. Minimal Axiom Systems Analysis",
        "",
        f"Found {len(theorem_groups)} distinct patterns of minimal axiom requirements.",
        "This shows how theorems cluster based on their logical dependencies.",
        "",
        "Most common minimal axiom systems:",
    ])
    
    for system, count in system_counts.most_common(5):
        report.append(f"- {system}: required for {count} theorems")
    
    report.extend([
        "",
        "## 4. Cycle Analysis",
        "",
        f"Identified {len(cycle_analysis)} fundamental cycles in the entailment graph.",
        "These represent alternative proof paths or circular dependencies in the logical structure.",
        "",
        "Cycle statistics:",
        f"- Average cycle length: {np.mean([c['length'] for c in cycle_analysis]):.2f} nodes",
        f"- Longest cycle: {max([c['length'] for c in cycle_analysis])} nodes",
        f"- Shortest cycle: {min([c['length'] for c in cycle_analysis])} nodes",
        "",
        "## 5. Research Implications",
        "",
        "### Structural Bottlenecks",
        "The cut vertices represent critical junctures in mathematical knowledge.",
        "These might be good targets for further research or axiomatization.",
        "",
        "### Logical Geography",
        "The connected components suggest natural divisions in mathematical reasoning.",
        "This may align with traditional divisions between different areas of mathematics.",
        "",
        "### Proof Economy",
        "The minimal axiom analysis reveals which formal systems are most economical",
        "for proving certain theorems. This could guide mathematicians toward the most",
        "efficient axiom systems for their work.",
        "",
        "### Redundancy and Robustness",
        "The fundamental cycles suggest redundancy in the axiom systems, providing",
        "multiple proof paths and making mathematical knowledge robust against the",
        "potential failure of individual axioms or proof techniques.",
    ])
    
    # Save report to markdown file
    filepath = os.path.join(OUTPUT_DIR, "entailment_analysis_report.md")
    with open(filepath, mode='w', encoding='utf-8') as file:
        file.write('\n'.join(report))
    
    print(f"Comprehensive analysis report saved to {filepath}")
    
    return report

if __name__ == "__main__":
    print("Loading entailment graph...")
    G = load_graph_from_csv()
    
    print("\nAnalyzing cut vertices...")
    cut_vertices, cut_vertex_analysis = analyze_cut_vertices(G)
    
    print("\nAnalyzing connected components...")
    components, component_analysis = analyze_components(G)
    
    print("\nAnalyzing minimal axiom systems...")
    theorem_groups, system_counts = analyze_minimal_axioms()
    
    print("\nAnalyzing fundamental cycles...")
    cycle_basis, cycle_analysis = analyze_cycles(G)
    
    print("\nGenerating comprehensive report...")
    generate_summary_report(cut_vertices, components, theorem_groups, system_counts, cycle_analysis)
    
    print("\nDeep analysis complete! All results saved to the entailment_output directory.")
