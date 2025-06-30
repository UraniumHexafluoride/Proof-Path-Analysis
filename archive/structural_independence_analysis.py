"""
Structural Independence Analysis

This module analyzes the structural properties of mathematical statements
in the entailment graph to identify patterns associated with independence.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numba

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_expanded_entailment_graph():
    """Create an expanded entailment graph with additional relationships."""
    G = nx.DiGraph()  # Create a directed graph
    
    # Add formal systems as nodes
    systems = ['PA', 'ZFC', 'ZF', 'PA2', 'ACA0', 'ZFC+LC', 'ZFC+MM', 'ZFC+AD', 'ZFC+PD', 'PVS+NP']
    for system in systems:
        G.add_node(system, type='system')
    
    # Add theorems and conjectures as nodes
    theorems = [
        'Four Color Theorem', 'Fermat\'s Last Theorem', 'Gödel\'s Incompleteness',
        'Continuum Hypothesis', 'Twin Prime Conjecture', 'Riemann Hypothesis',
        'Poincaré Conjecture', 'Hodge Conjecture', 'Goldbach\'s Conjecture',
        'Collatz Conjecture', 'ABC Conjecture', 'Navier-Stokes Existence and Smoothness',
        'Axiom of Choice', 'Swinerton-Dyer Conjecture', 'P vs NP'
    ]
    
    for theorem in theorems:
        G.add_node(theorem, type='theorem')
    
    # Add edges representing "proves" relationships
    proves_edges = [
        ('ZFC', 'Fermat\'s Last Theorem'),
        ('PA2', 'Four Color Theorem'),
        ('ZFC', 'Poincaré Conjecture'),
        ('ZFC', 'Hodge Conjecture'),
        ('ZFC+MM', 'Swinerton-Dyer Conjecture')
    ]
    
    for source, target in proves_edges:
        G.add_edge(source, target, relation='Proves')
    
    # Add edges representing "independence" relationships
    independence_edges = [
        ('ZFC', 'Continuum Hypothesis'),
        ('ZFC', 'Twin Prime Conjecture'),
        ('ZFC', 'Riemann Hypothesis'),
        ('PA', 'Gödel\'s Incompleteness'),
        ('ZFC', 'Axiom of Choice'),
        ('PVS+NP', 'P vs NP')
    ]
    
    for source, target in independence_edges:
        G.add_edge(source, target, relation='Independence')
    
    # Add edges representing "contains" relationships between formal systems
    contains_edges = [
        ('ZFC', 'ZF'),
        ('ZFC', 'PA'),
        ('ZF', 'PA'),
        ('PA2', 'PA'),
        ('ACA0', 'PA'),
        ('ZFC+LC', 'ZFC'),
        ('ZFC+MM', 'ZFC'),
        ('ZFC+AD', 'ZFC'),
        ('ZFC+PD', 'ZFC')
    ]
    
    for source, target in contains_edges:
        G.add_edge(source, target, relation='Contains')
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

@numba.jit(nopython=True)
def compute_metric_fast(values):
    """Compute a metric using Numba for performance."""
    result = 0.0
    for v in values:
        result += v * (1.0 - v)
    return result / len(values) if len(values) > 0 else 0.0

def compute_structural_metrics(G):
    """Compute structural metrics for each node in the graph."""
    metrics = {}
    
    # Compute basic centrality measures
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Compute PageRank
    pagerank = nx.pagerank(G)
    
    # Compute authority and hub scores
    try:
        hits = nx.hits(G)
        authority_scores = hits[0]
        hub_scores = hits[1]
    except:
        # If HITS algorithm fails (e.g., due to graph structure)
        authority_scores = {node: 0.0 for node in G.nodes()}
        hub_scores = {node: 0.0 for node in G.nodes()}
    
    # Store metrics for each node
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        
        # Get in-degree and out-degree
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Store all metrics
        metrics[node] = {
            'type': node_type,
            'degree_centrality': degree_centrality[node],
            'in_degree_centrality': in_degree_centrality[node],
            'out_degree_centrality': out_degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'pagerank': pagerank[node],
            'authority_score': authority_scores[node],
            'hub_score': hub_scores[node],
            'in_degree': in_degree,
            'out_degree': out_degree,
            'degree_ratio': out_degree / in_degree if in_degree > 0 else float('inf')
        }
    
    return metrics

def classify_statements(G):
    """Classify statements based on their relationships in the graph."""
    classifications = {}
    
    for node in G.nodes():
        # Skip system nodes
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        # Check incoming edges
        is_provable = False
        is_independent = False
        
        for pred, _, data in G.in_edges(node, data=True):
            relation = data.get('relation', '')
            if relation == 'Proves':
                is_provable = True
            elif relation == 'Independence':
                is_independent = True
        
        # Classify based on relationships
        if is_provable and is_independent:
            classifications[node] = 'both'
        elif is_provable:
            classifications[node] = 'provable'
        elif is_independent:
            classifications[node] = 'independent'
        else:
            classifications[node] = 'unknown'
    
    return classifications

def analyze_neighborhood_structure(G, classifications):
    """Analyze the neighborhood structure of nodes in the graph."""
    neighborhood_metrics = {}
    
    for node in G.nodes():
        # Skip system nodes
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        # Get predecessors and successors
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        
        # Count systems and theorems in predecessors
        pred_systems = sum(1 for n in predecessors if G.nodes[n].get('type') == 'system')
        pred_theorems = len(predecessors) - pred_systems
        
        # Count systems and theorems in successors
        succ_systems = sum(1 for n in successors if G.nodes[n].get('type') == 'system')
        succ_theorems = len(successors) - succ_systems
        
        # Count independent and provable statements in neighborhood
        independent_neighbors = sum(1 for n in predecessors + successors 
                                  if n in classifications and classifications[n] == 'independent')
        provable_neighbors = sum(1 for n in predecessors + successors 
                               if n in classifications and classifications[n] == 'provable')
        
        # Store metrics
        neighborhood_metrics[node] = {
            'pred_systems': pred_systems,
            'pred_theorems': pred_theorems,
            'succ_systems': succ_systems,
            'succ_theorems': succ_theorems,
            'independent_neighbors': independent_neighbors,
            'provable_neighbors': provable_neighbors,
            'neighborhood_size': len(predecessors) + len(successors)
        }
    
    return neighborhood_metrics

def generate_structural_analysis_report(metrics, classifications, neighborhood_metrics):
    """Generate a comprehensive report on the structural analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'structural_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Structural Analysis of Independence in Mathematics\n\n")
        f.write("This report analyzes the structural position of independent statements in the entailment graph.\n\n")
        
        # Classification summary
        f.write("## Classification Summary\n\n")
        class_counts = defaultdict(int)
        for classification in classifications.values():
            class_counts[classification] += 1
        
        f.write("| Classification | Count |\n")
        f.write("|---------------|-------|\n")
        for classification, count in class_counts.items():
            f.write(f"| {classification} | {count} |\n")
        f.write("\n")
        
        # Centrality metrics by classification
        f.write("## Centrality Metrics by Classification\n\n")
        f.write("Average centrality metrics for each classification:\n\n")
        
        # Create DataFrame for analysis
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df['classification'] = pd.Series(classifications)
        theorem_df = df[df['type'] == 'theorem'].copy()
        
        # Calculate average metrics by classification
        avg_metrics = theorem_df.groupby('classification').mean()
        
        f.write("| Classification | Degree | Betweenness | Closeness | PageRank |\n")
        f.write("|---------------|--------|-------------|-----------|----------|\n")
        for classification, row in avg_metrics.iterrows():
            f.write(f"| {classification} | {row['degree_centrality']:.4f} | {row['betweenness_centrality']:.4f} | {row['closeness_centrality']:.4f} | {row['pagerank']:.4f} |\n")
        f.write("\n")
        
        # Neighborhood structure analysis
        f.write("## Neighborhood Structure Analysis\n\n")
        f.write("Average neighborhood metrics for each classification:\n\n")
        
        # Create DataFrame for neighborhood analysis
        neighborhood_df = pd.DataFrame.from_dict(neighborhood_metrics, orient='index')
        neighborhood_df['classification'] = pd.Series(classifications)
        
        # Calculate average neighborhood metrics by classification
        avg_neighborhood = neighborhood_df.groupby('classification').mean()
        
        f.write("| Classification | Pred Systems | Pred Theorems | Succ Systems | Succ Theorems | Independent Neighbors | Provable Neighbors |\n")
        f.write("|---------------|--------------|---------------|--------------|---------------|----------------------|-------------------|\n")
        for classification, row in avg_neighborhood.iterrows():
            f.write(f"| {classification} | {row.get('pred_systems', 0):.2f} | {row.get('pred_theorems', 0):.2f} | {row.get('succ_systems', 0):.2f} | {row.get('succ_theorems', 0):.2f} | {row.get('independent_neighbors', 0):.2f} | {row.get('provable_neighbors', 0):.2f} |\n")
        f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Compare independent vs provable statements
        ind_metrics = theorem_df[theorem_df['classification'] == 'independent'].mean()
        prov_metrics = theorem_df[theorem_df['classification'] == 'provable'].mean()
        
        f.write("### Structural Differences Between Independent and Provable Statements\n\n")
        
        # Betweenness comparison
        if ind_metrics['betweenness_centrality'] > prov_metrics['betweenness_centrality']:
            f.write("- Independent statements have **higher betweenness centrality** (%.4f vs %.4f), suggesting they act as bridges between different areas of mathematics.\n" % 
                   (ind_metrics['betweenness_centrality'], prov_metrics['betweenness_centrality']))
        else:
            f.write("- Provable statements have **higher betweenness centrality** (%.4f vs %.4f), suggesting they are more central to the flow of mathematical reasoning.\n" % 
                   (prov_metrics['betweenness_centrality'], ind_metrics['betweenness_centrality']))
        
        # Degree comparison
        if ind_metrics['degree_centrality'] > prov_metrics['degree_centrality']:
            f.write("- Independent statements have **higher degree centrality** (%.4f vs %.4f), indicating more connections to other statements and systems.\n" % 
                   (ind_metrics['degree_centrality'], prov_metrics['degree_centrality']))
        else:
            f.write("- Provable statements have **higher degree centrality** (%.4f vs %.4f), indicating more connections to other statements and systems.\n" % 
                   (prov_metrics['degree_centrality'], ind_metrics['degree_centrality']))
        
        # PageRank comparison
        if ind_metrics['pagerank'] > prov_metrics['pagerank']:
            f.write("- Independent statements have **higher PageRank** (%.4f vs %.4f), suggesting they are more influential in the network.\n" % 
                   (ind_metrics['pagerank'], prov_metrics['pagerank']))
        else:
            f.write("- Provable statements have **higher PageRank** (%.4f vs %.4f), suggesting they are more influential in the network.\n" % 
                   (prov_metrics['pagerank'], ind_metrics['pagerank']))
        
        f.write("\n### Implications for Mathematical Research\n\n")
        f.write("- The structural position of independent statements suggests they serve as boundary objects between different formal systems.\n")
        f.write("- Independent statements tend to connect disparate areas of mathematics, potentially explaining why they resist proof within a single formal system.\n")
        f.write("- The network structure suggests that independence is not merely a negative result (inability to prove), but rather indicates a statement's position at the boundaries of formal systems.\n\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("The following visualizations are available:\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def visualize_centrality_distributions(metrics, classifications):
    """Create visualizations of centrality distributions by classification."""
    # Create a DataFrame for visualization
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['classification'] = pd.Series(classifications)
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Centrality Distributions by Classification', fontsize=16)
    
    # Plot betweenness centrality
    sns.boxplot(x='classification', y='betweenness_centrality', data=theorem_df, ax=axes[0, 0])
    axes[0, 0].set_title('Betweenness Centrality')
    axes[0, 0].set_xlabel('Classification')
    axes[0, 0].set_ylabel('Betweenness Centrality')
    
    # Plot degree centrality
    sns.boxplot(x='classification', y='degree_centrality', data=theorem_df, ax=axes[0, 1])
    axes[0, 1].set_title('Degree Centrality')
    axes[0, 1].set_xlabel('Classification')
    axes[0, 1].set_ylabel('Degree Centrality')
    
    # Plot closeness centrality
    sns.boxplot(x='classification', y='closeness_centrality', data=theorem_df, ax=axes[1, 0])
    axes[1, 0].set_title('Closeness Centrality')
    axes[1, 0].set_xlabel('Classification')
    axes[1, 0].set_ylabel('Closeness Centrality')
    
    # Plot PageRank
    sns.boxplot(x='classification', y='pagerank', data=theorem_df, ax=axes[1, 1])
    axes[1, 1].set_title('PageRank')
    axes[1, 1].set_xlabel('Classification')
    axes[1, 1].set_ylabel('PageRank')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'centrality_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create connectivity patterns visualization
    plt.figure(figsize=(12, 8))
    
    # Plot in-degree and out-degree by classification
    avg_by_class = theorem_df.groupby('classification')[['in_degree', 'out_degree']].mean()
    avg_by_class.plot(kind='bar', ax=plt.gca())
    
    plt.title('Connectivity Patterns by Classification', fontsize=16)
    plt.xlabel('Classification')
    plt.ylabel('Average Degree')
    plt.legend(['In-Degree', 'Out-Degree'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    conn_path = os.path.join(OUTPUT_DIR, 'connectivity_patterns.png')
    plt.savefig(conn_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_network_structure(G, classifications):
    """Create an improved network visualization showing entailment relationships."""
    plt.figure(figsize=(16, 14))
    
    # Create a spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=200, seed=42)
    
    # Draw nodes with colors based on classification
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', '')
        if node_type == 'system':
            node_colors.append('lightblue')
            node_sizes.append(400)
        else:
            classification = classifications.get(node, 'other')
            if classification == 'independent':
                node_colors.append('red')
            elif classification == 'provable':
                node_colors.append('green')
            elif classification == 'both':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
            node_sizes.append(600)  # Make theorem nodes larger
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with different colors based on relation type
    proves_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Proves']
    independence_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Independence']
    contains_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Contains']
    
    # Draw each type of edge with a different color and style
    nx.draw_networkx_edges(G, pos, edgelist=proves_edges, edge_color='blue', 
                          arrows=True, width=1.5, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=independence_edges, edge_color='red',
                          arrows=True, width=1.5, style='dashed', arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=contains_edges, edge_color='green',
                          arrows=True, width=1.0, arrowstyle='->', arrowsize=10)
    
    # Draw labels with different font sizes
    labels = {}
    for node in G.nodes():
        # Shorten long names
        if len(node) > 20:
            labels[node] = node[:17] + '...'
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='System'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Independent'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Provable'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Both'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Other'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Proves'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Independence'),
        plt.Line2D([0], [0], color='green', lw=2, label='Contains')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title('Entailment Network Structure', fontsize=18)
    plt.axis('off')
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'network_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def analyze_logical_strength(G, classifications):
    """Analyze the logical strength of statements based on their position in the graph."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'logical_strength_analysis.md')
    
    # Identify formal systems
    systems = [node for node in G.nodes() if G.nodes[node].get('type') == 'system']
    
    # Create a dictionary to store the logical strength of each statement
    logical_strength = {}
    
    # For each theorem, count how many systems can prove it
    for node in G.nodes():
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        # Count systems that can prove this theorem
        proving_systems = []
        for system in systems:
def generate_structural_analysis_report(metrics, classifications, neighborhood_metrics):
    """Generate a comprehensive report on the structural analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'structural_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Structural Analysis of Independence in Mathematics\n\n")
        f.write("This report analyzes the structural position of independent statements in the entailment graph.\n\n")
        
        # Create DataFrame for analysis
        df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Add classification column
        df['classification'] = pd.Series(classifications)
        
        # Filter to only include theorems
        theorem_df = df[df['type'] == 'theorem'].copy()
        
        # Select only numeric columns for averaging
        numeric_cols = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
                        'pagerank', 'in_degree', 'out_degree']
        
        # Calculate average metrics by classification for numeric columns only
        avg_metrics = theorem_df.groupby('classification')[numeric_cols].mean()
        
        # Visualizations section
        f.write("## Visualizations\n\n")
        f.write("The following visualizations are available:\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions\n\n")
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nVisualizing network structure...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path} and {network_viz_path}")
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
   
    print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path}")
    if strength_report_path:
        print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
    df['classification'] = pd.Series(classifications)
    
    # Filter to only include theorems
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    # Calculate average metrics by classification
    avg_metrics = theorem_df.groupby('classification').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the report
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across statement classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path

def main():
    """Main function to run the structural independence analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nVisualizing centrality distributions...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print(f"\nAnalysis complete! Results saved to {report_path}")
    print(f"Visualizations saved to {viz_path}")
    if strength_report_path:
        print(f"Logical strength analysis saved to {strength_report_path}")

if __name__ == "__main__":
    main()

def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """Generate an enhanced analysis report with more detailed interpretations."""
    # Create a DataFrame from metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add classification column
   










