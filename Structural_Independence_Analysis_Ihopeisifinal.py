import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Output directory for reports and figures
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_expanded_entailment_graph():
    """
    Create an expanded entailment graph with additional relationships.
    
    - Adds formal systems (nodes labeled as 'system')
    - Adds theorems/conjectures (nodes labeled as 'theorem')
    - Creates edges for "Proves", "Independence", and "Contains" relationships.
    
    Returns:
        G (nx.DiGraph): The constructed directed graph
    """
    G = nx.DiGraph()  # Initialize directed graph

    # Add formal systems as nodes
    systems = [
        'PA', 'ZFC', 'ZF', 'PA2', 'ACA0',
        'ZFC+LC', 'ZFC+MM', 'ZFC+AD', 'ZFC+PD', 'PVS+NP'
    ]
    for system in systems:
        G.add_node(system, type='system')

    # Add theorems and conjectures as nodes (with type 'theorem')
    theorems = [
        'Four Color Theorem', "Fermat's Last Theorem", "Gödel's Incompleteness",
        'Continuum Hypothesis', 'Twin Prime Conjecture', 'Riemann Hypothesis',
        'Poincaré Conjecture', 'Hodge Conjecture', "Goldbach's Conjecture",
        'Collatz Conjecture', 'ABC Conjecture', 'Navier-Stokes Existence and Smoothness',
        'Axiom of Choice', 'Swinerton-Dyer Conjecture', 'P vs NP'
    ]
    for theorem in theorems:
        G.add_node(theorem, type='theorem')

    # Add edges representing "Proves" relationships
    proves_edges = [
        ('ZFC', "Fermat's Last Theorem"),
        ('PA2', 'Four Color Theorem'),
        ('ZFC', 'Poincaré Conjecture'),
        ('ZFC', 'Hodge Conjecture'),
        ('ZFC+MM', 'Swinerton-Dyer Conjecture')
    ]
    for source, target in proves_edges:
        G.add_edge(source, target, relation='Proves')

    # Add edges representing "Independence" relationships
    independence_edges = [
        ('ZFC', 'Continuum Hypothesis'),
        ('ZFC', 'Twin Prime Conjecture'),
        ('ZFC', 'Riemann Hypothesis'),
        ('PA', "Gödel's Incompleteness"),
        ('ZFC', 'Axiom of Choice'),
        ('PVS+NP', 'P vs NP')
    ]
    for source, target in independence_edges:
        G.add_edge(source, target, relation='Independence')

    # Add edges representing "Contains" relationships (formal systems hierarchy)
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


def compute_structural_metrics(G):
    """
    Compute structural centrality metrics for each node in the graph.
    
    Calculates degree, betweenness, closeness, PageRank, and (if possible) HITS scores.
    Also records in-degree and out-degree counts.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        
    Returns:
        metrics (dict): Dictionary of metric values keyed by node.
    """
    metrics = {}
    
    # Compute various centrality measures.
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    
    # Try computing HITS scores; provide defaults if the algorithm fails.
    try:
        hits = nx.hits(G)
        authority_scores = hits[0]
        hub_scores = hits[1]
    except Exception as e:
        print(f"HITS computation failed: {e}")
        authority_scores = {node: 0.0 for node in G.nodes()}
        hub_scores = {node: 0.0 for node in G.nodes()}
    
    # Iterate over nodes to collect metrics.
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
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
    """
    Classify 'theorem' nodes based on incoming edge relationships.
    
    For each node with type 'theorem', determines whether it is 'provable', 
    'independent', 'both' (if both), or 'unknown' based on the edge relations.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        
    Returns:
        classifications (dict): Dictionary mapping node to classification string.
    """
    classifications = {}
    
    for node in G.nodes():
        # Only classify nodes that are theorems.
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        is_provable = False
        is_independent = False
        
        # Check incoming edges for relationship types.
        for pred, _, data in G.in_edges(node, data=True):
            relation = data.get('relation', '')
            if relation == 'Proves':
                is_provable = True
            elif relation == 'Independence':
                is_independent = True
        
        # Determine classification.
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
    """
    Analyze the local neighborhood of theorem nodes.
    
    Computes counts for predecessor and successor nodes, distinguishing between
    formal system nodes and theorem nodes, as well as counts of independent and 
    provable neighbors.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Mapping from node to its classification.
        
    Returns:
        neighborhood_metrics (dict): Neighborhood statistics for each theorem node.
    """
    neighborhood_metrics = {}
    
    for node in G.nodes():
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        
        # Count system nodes among neighbors.
        pred_systems = sum(1 for n in predecessors if G.nodes[n].get('type') == 'system')
        pred_theorems = len(predecessors) - pred_systems
        succ_systems = sum(1 for n in successors if G.nodes[n].get('type') == 'system')
        succ_theorems = len(successors) - succ_systems

        independent_neighbors = sum(1 for n in (predecessors + successors)
                                   if classifications.get(n) == 'independent')
        provable_neighbors = sum(1 for n in (predecessors + successors)
                                 if classifications.get(n) == 'provable')
        
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
    """
    Generate a comprehensive Markdown report of the structural analysis.
    
    The report includes a classification summary, average centrality metrics,
    and neighborhood structure analysis.
    
    Args:
        metrics (dict): Structural metrics computed for the graph.
        classifications (dict): Mapping of theorem nodes to classifications.
        neighborhood_metrics (dict): Neighborhood statistics.
        
    Returns:
        report_path (str): Path to the generated report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'structural_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Structural Analysis of Independence in Mathematics\n\n")
        f.write("This report analyzes the structural position of independent statements in the entailment graph.\n\n")
        
        # Classification summary.
        f.write("## Classification Summary\n\n")
        class_counts = defaultdict(int)
        for cls in classifications.values():
            class_counts[cls] += 1
        
        f.write("| Classification | Count |\n")
        f.write("|---------------|-------|\n")
        for cls, count in class_counts.items():
            f.write(f"| {cls} | {count} |\n")
        f.write("\n")
        
        # Average centrality metrics per classification.
        f.write("## Centrality Metrics by Classification\n\n")
        f.write("Average centrality metrics for each classification:\n\n")
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df['classification'] = pd.Series(classifications)
        theorem_df = df[df['type'] == 'theorem'].copy()
        numeric_cols = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']
        avg_metrics = theorem_df.groupby('classification')[numeric_cols].mean()
        
        f.write("| Classification | Degree | Betweenness | Closeness | PageRank |\n")
        f.write("|---------------|--------|-------------|-----------|----------|\n")
        for cls, row in avg_metrics.iterrows():
            f.write(f"| {cls} | {row['degree_centrality']:.4f} | {row['betweenness_centrality']:.4f} | "
                    f"{row['closeness_centrality']:.4f} | {row['pagerank']:.4f} |\n")
        f.write("\n")
        
        # Neighborhood structure analysis.
        f.write("## Neighborhood Structure Analysis\n\n")
        f.write("Average neighborhood metrics for each classification:\n\n")
        neighborhood_df = pd.DataFrame.from_dict(neighborhood_metrics, orient='index')
        neighborhood_df['classification'] = pd.Series(classifications)
        avg_neighborhood = neighborhood_df.groupby('classification').mean()
        
        f.write("| Classification | Pred Systems | Pred Theorems | Succ Systems | Succ Theorems | Independent Neighbors | Provable Neighbors |\n")
        f.write("|---------------|--------------|---------------|--------------|---------------|----------------------|-------------------|\n")
        for cls, row in avg_neighborhood.iterrows():
            f.write(f"| {cls} | {row.get('pred_systems', 0):.2f} | {row.get('pred_theorems', 0):.2f} | "
                    f"{row.get('succ_systems', 0):.2f} | {row.get('succ_theorems', 0):.2f} | "
                    f"{row.get('independent_neighbors', 0):.2f} | {row.get('provable_neighbors', 0):.2f} |\n")
        f.write("\n")
        
        # Key findings.
        f.write("## Key Findings\n\n")
        f.write("- The comparison of betweenness suggests a difference in how central independent or provable statements are.\n")
        f.write("- Degree and PageRank differences might indicate varying levels of influence within the network.\n")
        f.write("- Neighborhood metrics reveal potential clustering of certain classifications.\n\n")
        
        # Visualizations listing.
        f.write("## Visualizations\n\n")
        f.write("The following figures have been generated:\n\n")
        f.write("1. `centrality_distributions.png`\n")
        f.write("2. `connectivity_patterns.png`\n")
        f.write("3. `network_structure.png`\n")
    
    return report_path


def visualize_centrality_distributions(metrics, classifications):
    """
    Create visualizations of centrality measures per classification.
    
    Generates four box plots for betweenness, degree, closeness centralities, and PageRank.
    Also creates a bar chart for average in-degree and out-degree.
    
    Args:
        metrics (dict): Centrality metrics.
        classifications (dict): Mapping of nodes to classifications.
        
    Returns:
        output_path (str): Path to the centrality distributions figure.
    """
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['classification'] = pd.Series(classifications)
    theorem_df = df[df['type'] == 'theorem'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Centrality Distributions by Classification', fontsize=16)
    
    sns.boxplot(x='classification', y='betweenness_centrality', data=theorem_df, ax=axes[0, 0])
    axes[0, 0].set_title('Betweenness Centrality')
    axes[0, 0].set_xlabel('Classification')
    
    sns.boxplot(x='classification', y='degree_centrality', data=theorem_df, ax=axes[0, 1])
    axes[0, 1].set_title('Degree Centrality')
    axes[0, 1].set_xlabel('Classification')
    
    sns.boxplot(x='classification', y='closeness_centrality', data=theorem_df, ax=axes[1, 0])
    axes[1, 0].set_title('Closeness Centrality')
    axes[1, 0].set_xlabel('Classification')
    
    sns.boxplot(x='classification', y='pagerank', data=theorem_df, ax=axes[1, 1])
    axes[1, 1].set_title('PageRank')
    axes[1, 1].set_xlabel('Classification')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, 'centrality_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart for connectivity patterns
    plt.figure(figsize=(12, 8))
    avg_by_class = theorem_df.groupby('classification')[['in_degree', 'out_degree']].mean()
    avg_by_class.plot(kind='bar')
    plt.title('Connectivity Patterns by Classification', fontsize=16)
    plt.xlabel('Classification')
    plt.ylabel('Average Degree')
    plt.legend(['In-Degree', 'Out-Degree'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    conn_path = os.path.join(OUTPUT_DIR, 'connectivity_patterns.png')
    plt.savefig(conn_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return one generated visualization path
    return output_path


def visualize_network_structure(G, classifications):
    """
    Generate a network visualization of the entailment graph.
    
    Uses a spring layout and color-codes system nodes and theorem nodes based on classification.
    Different edge colors/styles denote different relationships.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Mapping of nodes to classifications.
        
    Returns:
        output_path (str): Path to the saved network structure figure.
    """
    if not G.nodes():
        print("Warning: Graph has no nodes. Visualization skipped.")
        return os.path.join(OUTPUT_DIR, 'network_structure.png')
    if not G.edges():
        print("Warning: Graph has no edges. Visualization may be incomplete.")

    plt.figure(figsize=(16, 14))
    pos = nx.spring_layout(G, k=0.5, iterations=200, seed=42)
    
    # Prepare node colors and sizes.
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', '')
        if node_type == 'system':
            node_colors.append('lightblue')
            node_sizes.append(400)
        else:
            cls = classifications.get(node, 'other')
            if cls == 'independent':
                node_colors.append('red')
            elif cls == 'provable':
                node_colors.append('green')
            elif cls == 'both':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
            node_sizes.append(600)
    
    # Draw the nodes.
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Separate edges by relation type.
    proves_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Proves']
    independence_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Independence']
    contains_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'Contains']
    
    nx.draw_networkx_edges(G, pos, edgelist=proves_edges, edge_color='blue',
                           arrows=True, width=1.5, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=independence_edges, edge_color='red',
                           arrows=True, width=1.5, style='dashed', arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=contains_edges, edge_color='green',
                           arrows=True, width=1.0, arrowstyle='->', arrowsize=10)
    
    # Add node labels.
    labels = {}
    for node in G.nodes():
        labels[node] = node if len(node) <= 20 else node[:17] + '...'
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Construct and draw legend.
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='System'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Independent'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Provable'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Both'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Other'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='solid', label='Proves'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Independence'),
        plt.Line2D([0], [0], color='green', lw=2, linestyle='solid', label='Contains')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title('Entailment Network Structure', fontsize=18)
    plt.axis('off')
    output_path = os.path.join(OUTPUT_DIR, 'network_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def analyze_logical_strength(G, classifications):
    """
    Analyze the logical strength of theorem nodes based on their graph positions.
    
    Computes strength based on the number of proving systems, proof power, and centrality.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Mapping of nodes to classifications.
        
    Returns:
        report_path (str): Path to the logical strength analysis report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'logical_strength_analysis.md')
    
    # Compute proof power: number of descendants (theorems provable from a node)
    proof_power = {}
    for node in G.nodes():
        descendants = set(nx.descendants(G, node))
        proof_power[node] = len([d for d in descendants if G.nodes[d].get('type') == 'theorem'])

    # Compute number of proving systems for theorems
    proving_systems = {}
    for node in G.nodes():
        if G.nodes[node].get('type') != 'theorem':
            continue
        proving_systems[node] = sum(1 for pred, _, data in G.in_edges(node, data=True)
                                   if data.get('relation') == 'Proves' and G.nodes[pred].get('type') == 'system')

    # Compute centrality-based strength (using PageRank)
    try:
        pagerank = nx.pagerank(G)
    except Exception as e:
        print(f"PageRank computation failed: {e}")
        pagerank = {node: 0.0 for node in G.nodes()}

    # Combine metrics into a strength score
    strength_scores = {}
    for node in G.nodes():
        if G.nodes[node].get('type') != 'theorem':
            continue
        # Weighted combination: 0.4 * proving systems, 0.3 * proof power, 0.3 * PageRank
        max_systems = max(proving_systems.values(), default=1) or 1
        max_power = max(proof_power.values(), default=1) or 1
        max_pagerank = max(pagerank.values(), default=1) or 1
        score = (0.4 * proving_systems.get(node, 0) / max_systems +
                 0.3 * proof_power.get(node, 0) / max_power +
                 0.3 * pagerank.get(node, 0) / max_pagerank)
        strength_scores[node] = score

    # Generate report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Logical Strength Analysis\n\n")
        f.write("This section analyzes the logical strength of theorems based on the number of "
                "formal systems that prove them, their proof power (number of theorems they imply), "
                "and their structural centrality (PageRank).\n\n")
        
        f.write("## Strength Scores by Theorem\n\n")
        f.write("| Theorem | Classification | Strength Score | Proving Systems | Proof Power | PageRank |\n")
        f.write("|---------|----------------|---------------|-----------------|-------------|----------|\n")
        for node in sorted(strength_scores.keys()):
            cls = classifications.get(node, 'unknown')
            f.write(f"| {node} | {cls} | {strength_scores[node]:.4f} | "
                    f"{proving_systems.get(node, 0)} | {proof_power.get(node, 0)} | "
                    f"{pagerank.get(node, 0):.4f} |\n")
        
        # Analyze strength by classification
        f.write("\n## Average Strength by Classification\n\n")
        df = pd.DataFrame({
            'theorem': list(strength_scores.keys()),
            'strength': list(strength_scores.values()),
            'classification': [classifications.get(n, 'unknown') for n in strength_scores]
        })
        avg_strength = df.groupby('classification')['strength'].mean()
        f.write("| Classification | Average Strength |\n")
        f.write("|---------------|------------------|\n")
        for cls, strength in avg_strength.items():
            f.write(f"| {cls} | {strength:.4f} |\n")
        
        f.write("\n## Findings\n\n")
        f.write("- Theorems with high strength scores are central to the entailment graph and have broad implications.\n")
        f.write("- Differences in strength by classification may indicate varying logical dependencies.\n")

    return report_path


def generate_enhanced_report(metrics, classifications, neighborhood_metrics):
    """
    Generate an enhanced Markdown report with extended interpretations and visualizations.
    
    Args:
        metrics (dict): Centrality and structural metrics.
        classifications (dict): Mapping of nodes to classifications.
        neighborhood_metrics (dict): Neighborhood statistics.
    
    Returns:
        report_path (str): Path to the enhanced analysis report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("This report provides extended interpretations of the structural analysis.\n\n")
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
    
    return report_path


def main():
    """
    Main function to run the complete structural independence analysis.
    
    Steps:
      1. Create the entailment graph.
      2. Compute structural metrics.
      3. Classify theorem nodes.
      4. Analyze neighborhood structure.
      5. Generate analysis reports.
      6. Create visualization figures.
    """
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nGenerating structural analysis report...")
    report_path = generate_structural_analysis_report(metrics, classifications, neighborhood_metrics)
    
    print("\nGenerating centrality visualizations...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nGenerating network structure visualization...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print("\nGenerating enhanced report...")
    enhanced_report_path = generate_enhanced_report(metrics, classifications, neighborhood_metrics)
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  Structural Analysis Report: {report_path}")
    print(f"  Centrality Visualizations: {viz_path}")
    print(f"  Network Structure Visualization: {network_viz_path}")
    print(f"  Logical Strength Analysis: {strength_report_path}")
    print(f"  Enhanced Report: {enhanced_report_path}")


if __name__ == "__main__":
    main()