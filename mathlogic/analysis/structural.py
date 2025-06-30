import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mathlogic.core.statements import get_all_theorems, get_all_systems, get_all_relationships
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Output directory for reports and figures
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_expanded_entailment_graph():
    """
    Create an expanded entailment graph with additional relationships.
    
    - Adds formal systems (nodes labeled as 'system')
    - Adds theorems/conjectures (nodes labeled as 'theorem')
    - Creates edges for "Proves", "Independence", and "Contains" relationships.
    - Includes intermediate theorems and theorem-to-theorem relationships
    
    Returns:
        G (nx.DiGraph): The constructed directed graph
    """
    G = nx.DiGraph()  # Initialize directed graph

    # Add formal systems as nodes
    systems_data = get_all_systems()
    for system_name in systems_data.keys():
        G.add_node(system_name, type='system')

    # Add theorems and conjectures as nodes (with type 'theorem')
    theorems_data = get_all_theorems()
    for theorem_name in theorems_data.keys():
        G.add_node(theorem_name, type='theorem')

    # Add edges based on relationships defined in statements.py
    all_relationships = get_all_relationships()
    
    # Separate relationships by type for easier processing
    proves_edges = []
    independence_edges = []
    contains_edges = []
    implies_edges = []

    for source, target, relation_type in all_relationships:
        if relation_type == 'proves':
            proves_edges.append((source, target))
        elif relation_type == 'independent':
            independence_edges.append((source, target))
        elif relation_type == 'contains':
            contains_edges.append((source, target))
        elif relation_type == 'implies':
            implies_edges.append((source, target))

    # Add edges representing "Proves" relationships
    # Existing hardcoded proves_edges are now handled by get_all_relationships
    # and the new structure in statements.py
    for source, target in proves_edges:
        G.add_edge(source, target, relation='Proves')

    # Add edges representing "Independence" relationships
    # Existing hardcoded independence_edges are now handled by get_all_relationships
    for source, target in independence_edges:
        G.add_edge(source, target, relation='Independence')

    # Add edges representing "Contains" relationships (formal systems hierarchy)
    # Existing hardcoded contains_edges are now handled by get_all_relationships
    for source, target in contains_edges:
        G.add_edge(source, target, relation='Contains')
        
    # Add theorem-to-theorem implications (creates paths through the graph)
    # Existing hardcoded implies_edges are now handled by get_all_relationships
    for source, target in implies_edges:
        G.add_edge(source, target, relation='Implies')

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
    
    Args:
        G (nx.DiGraph): The entailment graph.
        
    Returns:
        classifications (dict): Dictionary mapping node to classification string.
    """
    classifications = {}
    theorem_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'theorem']
    
    for node in theorem_nodes:
        is_provable = False
        is_independent = False
        
        # Check incoming edges in one pass
        for _, _, data in G.in_edges(node, data=True):
            relation = data.get('relation', '')
            if relation == 'Proves':
                is_provable = True
            elif relation == 'Independence':
                is_independent = True
            
            # Early exit if both conditions are met
            if is_provable and is_independent:
                break
        
        # Determine classification
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
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Mapping from node to its classification.
        
    Returns:
        neighborhood_metrics (dict): Neighborhood statistics for each theorem node.
    """
    neighborhood_metrics = {}
    theorem_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'theorem']
    
    for node in theorem_nodes:
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        all_neighbors = predecessors + successors
        
        # Count in one pass
        counts = {
            'pred_systems': 0,
            'pred_theorems': 0,
            'succ_systems': 0,
            'succ_theorems': 0,
            'independent_neighbors': 0,
            'provable_neighbors': 0
        }
        
        for n in predecessors:
            if G.nodes[n].get('type') == 'system':
                counts['pred_systems'] += 1
            else:
                counts['pred_theorems'] += 1
        
        for n in successors:
            if G.nodes[n].get('type') == 'system':
                counts['succ_systems'] += 1
            else:
                counts['succ_theorems'] += 1
        
        for n in all_neighbors:
            if classifications.get(n) == 'independent':
                counts['independent_neighbors'] += 1
            elif classifications.get(n) == 'provable':
                counts['provable_neighbors'] += 1
        
        counts['neighborhood_size'] = len(all_neighbors)
        neighborhood_metrics[node] = counts
    
    return neighborhood_metrics


def analyze_extended_neighborhood(G, classifications, max_hops=2):
    """
    Analyze the extended neighborhood structure (up to max_hops away) for theorem nodes.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Mapping from node to its classification.
        max_hops (int): Maximum distance to analyze.
        
    Returns:
        extended_metrics (dict): Extended neighborhood statistics for each theorem node.
    """
    extended_metrics = {}
    
    # Get only theorem nodes
    theorem_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'theorem']
    # Create undirected version once for multi-hop analysis
    G_undir = G.to_undirected()
    
    for node in theorem_nodes:
        node_metrics = {}
        
        # For each hop distance
        for hop in range(1, max_hops + 1):
            # Get nodes at exactly distance 'hop'
            if hop == 1:
                # Direct neighbors (predecessors and successors)
                neighbors = set(G.predecessors(node)).union(set(G.successors(node)))
            else:
                # Get all nodes up to distance 'hop'
                all_within_hop = set(nx.single_source_shortest_path_length(G_undir, node, cutoff=hop).keys())
                # Get all nodes up to distance 'hop-1'
                all_within_prev_hop = set(nx.single_source_shortest_path_length(G_undir, node, cutoff=hop-1).keys())
                # Nodes exactly at distance 'hop' are the difference
                neighbors = all_within_hop - all_within_prev_hop
            
            # Count node types and classifications in one pass
            counts = {
                'systems_count': 0,
                'theorems_count': 0,
                'independent_count': 0,
                'provable_count': 0,
                'both_count': 0,
                'unknown_count': 0
            }
            
            for n in neighbors:
                # Count by node type
                if G.nodes[n].get('type') == 'system':
                    counts['systems_count'] += 1
                elif G.nodes[n].get('type') == 'theorem':
                    counts['theorems_count'] += 1
                
                # Count by classification
                if n in classifications:
                    cls = classifications[n]
                    counts[f'{cls}_count'] += 1
            
            # Store metrics for this hop distance
            node_metrics[f'hop_{hop}'] = counts
        
        extended_metrics[node] = node_metrics
    
    return extended_metrics


def generate_structural_analysis_report(metrics, classifications, neighborhood_metrics, extended_metrics=None):
    """
    Generate a comprehensive Markdown report of the structural analysis.
    
    The report includes a classification summary, average centrality metrics,
    neighborhood structure analysis, and extended neighborhood analysis.
    
    Args:
        metrics (dict): Structural metrics computed for the graph.
        classifications (dict): Mapping of theorem nodes to classifications.
        neighborhood_metrics (dict): Neighborhood statistics.
        extended_metrics (dict, optional): Extended neighborhood statistics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'structural_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Report header and introduction
        f.write("# Structural Independence Analysis\n\n")
        f.write("This report analyzes the structural properties of mathematical statements ")
        f.write("and their relationships to formal systems.\n\n")
        
        # Classification summary
        f.write("## Classification Summary\n\n")
        class_counts = Counter(classifications.values())
        f.write(f"- Provable statements: {class_counts.get('provable', 0)}\n")
        f.write(f"- Independent statements: {class_counts.get('independent', 0)}\n")
        f.write(f"- Both (provable in some systems, independent in others): {class_counts.get('both', 0)}\n")
        f.write(f"- Unknown classification: {class_counts.get('unknown', 0)}\n\n")
        
        # Average centrality metrics by classification
        f.write("## Centrality Metrics by Classification\n\n")
        f.write("Average centrality metrics for each classification:\n\n")
        
        # Create a DataFrame for analysis
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df['classification'] = pd.Series(classifications)
        theorem_df = df[df['type'] == 'theorem'].copy()
        
        # Select only numeric columns for averaging
        numeric_cols = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
                        'betweenness_centrality', 'closeness_centrality', 'pagerank', 
                        'authority_score', 'hub_score', 'in_degree', 'out_degree']
        
        # Make sure we only include columns that actually exist
        numeric_cols = [col for col in numeric_cols if col in theorem_df.columns]
        
        # Calculate means for numeric columns only
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
        
        # Select only numeric columns for neighborhood metrics
        neigh_numeric_cols = [col for col in neighborhood_df.columns 
                             if col != 'classification' and neighborhood_df[col].dtype in ['int64', 'float64']]
        
        avg_neighborhood = neighborhood_df.groupby('classification')[neigh_numeric_cols].mean()
        
        f.write("| Classification | Neighborhood Size | System Predecessors | Theorem Predecessors | System Successors | Theorem Successors |\n")
        f.write("|---------------|-------------------|---------------------|---------------------|------------------|-------------------|\n")
        for cls, row in avg_neighborhood.iterrows():
            f.write(f"| {cls} | {row['neighborhood_size']:.2f} | {row['pred_systems']:.2f} | "
                    f"{row['pred_theorems']:.2f} | {row['succ_systems']:.2f} | {row['succ_theorems']:.2f} |\n")
        f.write("\n")
        
        # Extended neighborhood analysis (if available)
        if extended_metrics:
            f.write("## Extended Neighborhood Analysis\n\n")
            f.write("Analysis of nodes at different hop distances from each theorem.\n\n")
            
            # Group data manually
            hop_metrics = {}
            for node, node_data in extended_metrics.items():
                cls = classifications.get(node, 'unknown')
                for hop in range(1, 3):  # Assuming max_hops=2
                    hop_key = f'hop_{hop}'
                    if hop_key in node_data:
                        group_key = (cls, hop)
                        if group_key not in hop_metrics:
                            hop_metrics[group_key] = {
                                'systems_count': [],
                                'theorems_count': [],
                                'independent_count': [],
                                'provable_count': [],
                                'total_neighbors': []
                            }
                        
                        hop_metrics[group_key]['systems_count'].append(node_data[hop_key]['systems_count'])
                        hop_metrics[group_key]['theorems_count'].append(node_data[hop_key]['theorems_count'])
                        hop_metrics[group_key]['independent_count'].append(node_data[hop_key]['independent_count'])
                        hop_metrics[group_key]['provable_count'].append(node_data[hop_key]['provable_count'])
                        
                        # Calculate total neighbors as sum of systems and theorems
                        total = node_data[hop_key]['systems_count'] + node_data[hop_key]['theorems_count']
                        hop_metrics[group_key]['total_neighbors'].append(total)
            
            # Write the table
            f.write("### Average Neighborhood Composition by Hop Distance\n\n")
            f.write("| Classification | Hop | Systems | Theorems | Independent | Provable | Total |\n")
            f.write("|---------------|-----|---------|----------|-------------|----------|-------|\n")
            
            for (cls, hop), counts in hop_metrics.items():
                systems_avg = sum(counts['systems_count']) / len(counts['systems_count']) if counts['systems_count'] else 0
                theorems_avg = sum(counts['theorems_count']) / len(counts['theorems_count']) if counts['theorems_count'] else 0
                independent_avg = sum(counts['independent_count']) / len(counts['independent_count']) if counts['independent_count'] else 0
                provable_avg = sum(counts['provable_count']) / len(counts['provable_count']) if counts['provable_count'] else 0
                total_avg = sum(counts['total_neighbors']) / len(counts['total_neighbors']) if counts['total_neighbors'] else 0
                
                f.write(f"| {cls} | {hop} | {systems_avg:.2f} | {theorems_avg:.2f} | "
                        f"{independent_avg:.2f} | {provable_avg:.2f} | {total_avg:.2f} |\n")
            
            f.write("\n")
        
        # Key findings section.
        f.write("## Key Findings\n\n")
        f.write("- The comparison of betweenness suggests a difference in how central independent or provable statements are.\n")
        f.write("- Degree and PageRank differences might indicate varying levels of influence within the network.\n")
        f.write("- Neighborhood metrics reveal potential clustering of certain classifications.\n")
        
        # Visualizations section.
        f.write("\n## Visualizations\n\n")
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


def generate_enhanced_report(metrics, classifications, neighborhood_metrics, extended_metrics=None):
    """
    Generate an enhanced Markdown report with extended interpretations and visualizations.
    
    Args:
        metrics (dict): Centrality and structural metrics.
        classifications (dict): Mapping of nodes to classifications.
        neighborhood_metrics (dict): Neighborhood statistics.
        extended_metrics (dict, optional): Extended neighborhood statistics.
    
    Returns:
        report_path (str): Path to the enhanced analysis report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'enhanced_independence_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Independence Analysis\n\n")
        f.write("This report provides extended interpretations of the structural analysis.\n\n")
        
        # Add interpretation of centrality metrics
        f.write("## Interpretation of Centrality Metrics\n\n")
        f.write("Centrality metrics reveal the structural importance of theorems in the mathematical landscape:\n\n")
        
        # Create a DataFrame for analysis
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df['classification'] = pd.Series(classifications)
        theorem_df = df[df['type'] == 'theorem'].copy()
        
        # Calculate average metrics by classification
        avg_metrics = theorem_df.groupby('classification')[
            ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']
        ].mean()
        
        # Interpret degree centrality
        f.write("### Degree Centrality\n\n")
        f.write("Degree centrality measures how many direct connections a theorem has.\n\n")
        
        # Find classification with highest degree centrality
        max_degree_cls = avg_metrics['degree_centrality'].idxmax()
        f.write(f"- **{max_degree_cls}** theorems have the highest average degree centrality " 
                f"({avg_metrics.loc[max_degree_cls, 'degree_centrality']:.4f}), suggesting they have more " 
                f"connections to other mathematical statements.\n")
        
        # Interpret betweenness centrality
        f.write("\n### Betweenness Centrality\n\n")
        f.write("Betweenness centrality measures how often a theorem acts as a bridge between other theorems.\n\n")
        
        # Find classification with highest betweenness centrality
        max_between_cls = avg_metrics['betweenness_centrality'].idxmax()
        f.write(f"- **{max_between_cls}** theorems have the highest average betweenness centrality " 
                f"({avg_metrics.loc[max_between_cls, 'betweenness_centrality']:.4f}), suggesting they " 
                f"serve as important bridges in mathematical reasoning.\n")
        
        # Interpret closeness centrality
        f.write("\n### Closeness Centrality\n\n")
        f.write("Closeness centrality measures how close a theorem is to all other theorems in the network.\n\n")
        
        # Find classification with highest closeness centrality
        max_close_cls = avg_metrics['closeness_centrality'].idxmax()
        f.write(f"- **{max_close_cls}** theorems have the highest average closeness centrality " 
                f"({avg_metrics.loc[max_close_cls, 'closeness_centrality']:.4f}), suggesting they are " 
                f"more central to the overall structure of mathematics.\n")
        
        # Interpret PageRank
        f.write("\n### PageRank\n\n")
        f.write("PageRank measures the global importance of a theorem based on the importance of its neighbors.\n\n")
        
        # Find classification with highest PageRank
        max_pr_cls = avg_metrics['pagerank'].idxmax()
        f.write(f"- **{max_pr_cls}** theorems have the highest average PageRank " 
                f"({avg_metrics.loc[max_pr_cls, 'pagerank']:.4f}), suggesting they are " 
                f"more influential in the mathematical landscape.\n")
        
        # Add neighborhood structure interpretation
        f.write("\n## Interpretation of Neighborhood Structure\n\n")
        f.write("The neighborhood structure reveals how theorems relate to their immediate surroundings:\n\n")
        
        # Create a DataFrame for neighborhood analysis
        neighborhood_df = pd.DataFrame.from_dict(neighborhood_metrics, orient='index')
        neighborhood_df['classification'] = pd.Series(classifications)
        
        # Calculate average neighborhood metrics by classification
        avg_neighborhood = neighborhood_df.groupby('classification').mean()
        
        # Interpret predecessor systems
        f.write("### Formal System Dependencies\n\n")
        
        # Find classification with most predecessor systems
        if 'pred_systems' in avg_neighborhood.columns:
            max_pred_sys_cls = avg_neighborhood['pred_systems'].idxmax()
            f.write(f"- **{max_pred_sys_cls}** theorems are proven by more formal systems " 
                    f"({avg_neighborhood.loc[max_pred_sys_cls, 'pred_systems']:.2f} on average), " 
                    f"suggesting they are more fundamental or widely accepted.\n")
        
        # Interpret neighborhood diversity
        f.write("\n### Neighborhood Diversity\n\n")
        
        if extended_metrics:
            # Calculate average diversity by classification
            diversity_by_class = {}
            for node, node_data in extended_metrics.items():
                cls = classifications.get(node, 'unknown')
                if cls not in diversity_by_class:
                    diversity_by_class[cls] = []
                
                if 'shannon_diversity' in node_data:
                    diversity_by_class[cls].append(node_data['shannon_diversity'])
            
            # Find classification with highest diversity
            max_diversity_cls = None
            max_diversity_val = 0
            for cls, diversity_values in diversity_by_class.items():
                avg_diversity = sum(diversity_values) / len(diversity_values) if diversity_values else 0
                if avg_diversity > max_diversity_val:
                    max_diversity_val = avg_diversity
                    max_diversity_cls = cls
            
            if max_diversity_cls:
                f.write(f"- **{max_diversity_cls}** theorems have the most diverse neighborhoods " 
                        f"(Shannon diversity index: {max_diversity_val:.4f}), suggesting they " 
                        f"connect different areas of mathematics.\n")
        
        # Add structural patterns section
        f.write("\n## Structural Patterns Associated with Independence\n\n")
        f.write("Based on our analysis, we can identify several structural patterns that are associated with independent statements:\n\n")
        
        # Compare independent vs. provable theorems
        if 'independent' in avg_metrics.index and 'provable' in avg_metrics.index:
            # Degree centrality comparison
            ind_degree = avg_metrics.loc['independent', 'degree_centrality']
            prov_degree = avg_metrics.loc['provable', 'degree_centrality']
            
            if ind_degree > prov_degree:
                f.write("1. **Higher Connectivity**: Independent statements tend to have higher degree centrality " 
                        f"({ind_degree:.4f} vs. {prov_degree:.4f}), suggesting they connect to more diverse parts of mathematics.\n\n")
            else:
                f.write("1. **Lower Connectivity**: Independent statements tend to have lower degree centrality " 
                        f"({ind_degree:.4f} vs. {prov_degree:.4f}), suggesting they are more isolated in the mathematical landscape.\n\n")
            
            # PageRank comparison
            ind_pr = avg_metrics.loc['independent', 'pagerank']
            prov_pr = avg_metrics.loc['provable', 'pagerank']
            
            if ind_pr > prov_pr:
                f.write("2. **Higher Influence**: Independent statements tend to have higher PageRank " 
                        f"({ind_pr:.4f} vs. {prov_pr:.4f}), suggesting they are more influential despite being unprovable.\n\n")
            else:
                f.write("2. **Lower Influence**: Independent statements tend to have lower PageRank " 
                        f"({ind_pr:.4f} vs. {prov_pr:.4f}), suggesting they are less central to mathematical reasoning.\n\n")
        
        # Add visualizations section
        f.write("## Available Visualizations\n\n")
        f.write("1. `centrality_distributions.png`: Box plots showing the distribution of centrality measures across classifications\n")
        f.write("2. `connectivity_patterns.png`: Bar charts showing in-degree and out-degree patterns by classification\n")
        f.write("3. `network_structure.png`: Network visualization of the entailment graph with nodes colored by classification\n")
        
        # Add conclusions section
        f.write("\n## Conclusions and Next Steps\n\n")
        f.write("Our analysis reveals several structural patterns associated with independence in mathematics:\n\n")
        f.write("1. Independent statements show distinctive centrality patterns compared to provable statements.\n")
        f.write("2. The neighborhood structure around independent statements differs from that of provable statements.\n")
        f.write("3. These structural differences could potentially be used to predict independence.\n\n")
        f.write("Next steps in this research include:\n\n")
        f.write("1. Expanding the graph with more theorems and relationships.\n")
        f.write("2. Developing a predictive model based on the identified structural patterns.\n")
        f.write("3. Testing the model on known independent statements to validate its accuracy.\n")
    
    return report_path


def compute_advanced_metrics(G, classifications):
    """
    Compute advanced structural metrics that may correlate with independence.
    
    Args:
        G (nx.DiGraph): The entailment graph
        classifications (dict): Known classifications of theorems
        
    Returns:
        dict: Advanced metrics for each theorem node
    """
    advanced_metrics = {}
    
    # Get all theorem nodes
    theorem_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'theorem']
    
    for node in theorem_nodes:
        node_metrics = {}
        
        # 1. System diversity: number of different formal systems that connect to this theorem
        predecessors = list(G.predecessors(node))
        system_predecessors = [p for p in predecessors if G.nodes[p].get('type') == 'system']
        node_metrics['system_diversity'] = len(system_predecessors)
        
        # 2. Independence ratio: ratio of independence edges to total edges for systems
        independence_edges = sum(1 for _, _, data in G.in_edges(node, data=True) 
                               if data.get('relation') == 'Independence')
        proves_edges = sum(1 for _, _, data in G.in_edges(node, data=True) 
                          if data.get('relation') == 'Proves')
        total_system_edges = independence_edges + proves_edges
        node_metrics['independence_ratio'] = independence_edges / total_system_edges if total_system_edges > 0 else 0
        
        # 3. Theorem neighborhood similarity: how similar is this theorem to known independent theorems
        independent_theorems = [t for t in theorem_nodes if classifications.get(t) == 'independent']
        provable_theorems = [t for t in theorem_nodes if classifications.get(t) == 'provable']
        
        # Jaccard similarity with independent theorems
        if independent_theorems:
            ind_similarities = []
            for ind_theorem in independent_theorems:
                ind_neighbors = set(G.neighbors(ind_theorem))
                node_neighbors = set(G.neighbors(node))
                if ind_neighbors or node_neighbors:
                    similarity = len(ind_neighbors & node_neighbors) / len(ind_neighbors | node_neighbors)
                    ind_similarities.append(similarity)
            node_metrics['independent_similarity'] = sum(ind_similarities) / len(ind_similarities) if ind_similarities else 0
        else:
            node_metrics['independent_similarity'] = 0
            
        # Jaccard similarity with provable theorems
        if provable_theorems:
            prov_similarities = []
            for prov_theorem in provable_theorems:
                prov_neighbors = set(G.neighbors(prov_theorem))
                node_neighbors = set(G.neighbors(node))
                if prov_neighbors or node_neighbors:
                    similarity = len(prov_neighbors & node_neighbors) / len(prov_neighbors | node_neighbors)
                    prov_similarities.append(similarity)
            node_metrics['provable_similarity'] = sum(prov_similarities) / len(prov_similarities) if prov_similarities else 0
        else:
            node_metrics['provable_similarity'] = 0
            
        # 4. Path diversity: number of distinct paths to this theorem
        node_metrics['path_diversity'] = 0
        for system in [n for n in G.nodes() if G.nodes[n].get('type') == 'system']:
            try:
                paths = list(nx.all_simple_paths(G, system, node, cutoff=5))
                node_metrics['path_diversity'] += len(paths)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
                
        # 5. Structural balance: ratio of triangles to wedges in neighborhood
        try:
            triangles = sum(1 for _ in nx.triangles(G.to_undirected(), [node]).values())
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            node_metrics['structural_balance'] = triangles / possible_triangles if possible_triangles > 0 else 0
        except:
            node_metrics['structural_balance'] = 0
        
        advanced_metrics[node] = node_metrics
    
    return advanced_metrics

def analyze_independence_patterns(G, classifications, metrics, advanced_metrics):
    """
    Analyze patterns that distinguish independent from provable theorems.
    
    Args:
        G (nx.DiGraph): The entailment graph
        classifications (dict): Known classifications of theorems
        metrics (dict): Basic structural metrics
        advanced_metrics (dict): Advanced structural metrics
        
    Returns:
        dict: Analysis results
    """
    results = {
        'independent_theorems': [],
        'provable_theorems': [],
        'metric_differences': {},
        'significant_indicators': []
    }
    
    # Separate theorems by classification
    for node, cls in classifications.items():
        if G.nodes[node].get('type') == 'theorem':
            if cls == 'independent':
                results['independent_theorems'].append(node)
            elif cls == 'provable':
                results['provable_theorems'].append(node)
    
    # Compare metrics between independent and provable theorems
    all_metrics = {}
    for node in G.nodes():
        if G.nodes[node].get('type') == 'theorem':
            all_metrics[node] = {
                **{k: v for k, v in metrics[node].items() if k != 'type'},
                **advanced_metrics.get(node, {})
            }
    
    # Calculate average metrics for each classification
    independent_avg = {metric: 0 for metric in all_metrics.get(results['independent_theorems'][0], {})} if results['independent_theorems'] else {}
    provable_avg = {metric: 0 for metric in all_metrics.get(results['provable_theorems'][0], {})} if results['provable_theorems'] else {}
    
    for node in results['independent_theorems']:
        for metric, value in all_metrics.get(node, {}).items():
            independent_avg[metric] = independent_avg.get(metric, 0) + value
    
    for node in results['provable_theorems']:
        for metric, value in all_metrics.get(node, {}).items():
            provable_avg[metric] = provable_avg.get(metric, 0) + value
    
    # Calculate averages
    for metric in independent_avg:
        independent_avg[metric] /= len(results['independent_theorems']) if results['independent_theorems'] else 1
    
    for metric in provable_avg:
        provable_avg[metric] /= len(results['provable_theorems']) if results['provable_theorems'] else 1
    
    # Calculate differences and identify significant indicators
    for metric in independent_avg:
        if metric in provable_avg:
            ind_val = independent_avg[metric]
            prov_val = provable_avg[metric]
            diff = ind_val - prov_val
            results['metric_differences'][metric] = diff
            
            # Identify metrics with substantial differences
            if abs(diff) > 0.1 and max(ind_val, prov_val) > 0:
                relative_diff = abs(diff) / max(ind_val, prov_val)
                if relative_diff > 0.2:  # 20% difference threshold
                    results['significant_indicators'].append({
                        'metric': metric,
                        'independent_avg': ind_val,
                        'provable_avg': prov_val,
                        'difference': diff,
                        'relative_difference': relative_diff
                    })
    
    # Sort significant indicators by relative difference
    results['significant_indicators'].sort(key=lambda x: abs(x['relative_difference']), reverse=True)
    
    return results

def generate_independence_patterns_report(patterns, output_path=None):
    """
    Generate a report on independence patterns.
    
    Args:
        patterns (dict): Results from analyze_independence_patterns
        output_path (str, optional): Path to save the report
        
    Returns:
        str: Path to the saved report
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'independence_patterns_report.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Independence Patterns Analysis\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This analysis compares {len(patterns['independent_theorems'])} independent theorems ")
        f.write(f"with {len(patterns['provable_theorems'])} provable theorems to identify structural patterns.\n\n")
        
        f.write("## Significant Indicators of Independence\n\n")
        
        if patterns['significant_indicators']:
            f.write("The following metrics show substantial differences between independent and provable theorems:\n\n")
            f.write("| Metric | Independent Avg | Provable Avg | Difference | Relative Difference |\n")
            f.write("|--------|----------------|--------------|------------|--------------------|\n")
            
            for indicator in patterns['significant_indicators']:
                f.write(f"| {indicator['metric']} | {indicator['independent_avg']:.4f} | ")
                f.write(f"{indicator['provable_avg']:.4f} | {indicator['difference']:.4f} | ")
                f.write(f"{indicator['relative_difference']:.2%} |\n")
        else:
            f.write("No significant indicators were found. This may be due to insufficient data or lack of clear patterns.\n")
        
        f.write("\n## Interpretation\n\n")
        
        if patterns['significant_indicators']:
            top_indicator = patterns['significant_indicators'][0]
            metric_name = top_indicator['metric']
            
            f.write(f"The most significant indicator is **{metric_name}**, where ")
            
            if top_indicator['difference'] > 0:
                f.write(f"independent theorems have a {top_indicator['relative_difference']:.2%} higher value than provable theorems.\n\n")
            else:
                f.write(f"provable theorems have a {abs(top_indicator['relative_difference']):.2%} higher value than independent theorems.\n\n")
            
            # Add interpretations for common metrics
            if metric_name == 'betweenness_centrality':
                f.write("Higher betweenness centrality for independent theorems suggests they act as bridges between different areas of mathematics.\n")
            elif metric_name == 'pagerank':
                f.write("Higher PageRank for independent theorems suggests they are more central to the mathematical landscape despite being unprovable.\n")
            elif metric_name == 'system_diversity':
                f.write("Higher system diversity for independent theorems suggests they connect to more diverse formal systems.\n")
            elif metric_name == 'independence_ratio':
                f.write("Higher independence ratio indicates that theorems with more independence relationships tend to be independent themselves.\n")
            elif metric_name == 'path_diversity':
                f.write("Higher path diversity suggests that independent theorems have more complex relationships with formal systems.\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on this analysis, we recommend:\n\n")
        f.write("1. Using these indicators to develop a predictive model for independence\n")
        f.write("2. Expanding the dataset with more theorems to strengthen the statistical significance\n")
        f.write("3. Investigating the causal relationships behind these correlations\n")
    
    return output_path


class StructuralAnalyzer:
    """Analyzes structural properties of the entailment graph."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the analyzer."""
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze the graph structure.
        
        Args:
            G: The entailment graph to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        # Compute basic structural metrics
        metrics = compute_structural_metrics(G)
        
        # Classify statements
        classifications = classify_statements(G)
        
        # Analyze neighborhood structure
        neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
        
        # Analyze extended neighborhood
        extended_metrics = analyze_extended_neighborhood(G, classifications)
        
        # Analyze logical strength
        strength_metrics = analyze_logical_strength(G, classifications)
        
        # Compute advanced metrics
        advanced_metrics = compute_advanced_metrics(G, classifications)
        
        # Analyze independence patterns
        patterns = analyze_independence_patterns(G, classifications, metrics, advanced_metrics)
        
        # Generate reports
        self._generate_reports(G, metrics, classifications, neighborhood_metrics,
                             extended_metrics, strength_metrics, patterns)
        
        # Generate visualizations
        self._generate_visualizations(G, metrics, classifications)
        
        # Return all metrics
        return {
            'structural_metrics': metrics,
            'classifications': classifications,
            'neighborhood_metrics': neighborhood_metrics,
            'extended_metrics': extended_metrics,
            'strength_metrics': strength_metrics,
            'advanced_metrics': advanced_metrics,
            'independence_patterns': patterns
        }
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str) -> None:
        """Save metrics to a JSON file."""
        def convert_to_serializable(obj):
            """Convert non-serializable objects to serializable ones."""
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        def process_dict(d):
            """Process a dictionary to make it JSON serializable."""
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = process_dict(v)
                elif isinstance(v, (list, tuple)):
                    result[k] = [convert_to_serializable(x) for x in v]
                else:
                    result[k] = convert_to_serializable(v)
            return result
        
        # Process metrics to ensure they're JSON serializable
        serializable_metrics = process_dict(metrics)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def _generate_reports(self, G, metrics, classifications, neighborhood_metrics,
                         extended_metrics, strength_metrics, patterns):
        """Generate all analysis reports."""
        # Generate structural analysis report
        generate_structural_analysis_report(
            metrics, classifications, neighborhood_metrics, extended_metrics
        )
        
        # Generate enhanced report
        generate_enhanced_report(
            metrics, classifications, neighborhood_metrics, extended_metrics
        )
        
        # Generate independence patterns report
        generate_independence_patterns_report(patterns)
    
    def _generate_visualizations(self, G, metrics, classifications):
        """Generate all visualizations."""
        visualize_centrality_distributions(metrics, classifications)
        visualize_network_structure(G, classifications)


def main():
    """
    Main function to run the complete structural independence analysis.
    
    Steps:
      1. Create the entailment graph.
      2. Compute structural metrics.
      3. Classify theorem nodes.
      4. Analyze neighborhood structure.
      5. Analyze extended neighborhood structure.
      6. Generate analysis reports.
      7. Create visualization figures.
    """
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nComputing structural metrics...")
    metrics = compute_structural_metrics(G)
    
    print("\nClassifying statements...")
    classifications = classify_statements(G)
    
    print("\nAnalyzing neighborhood structure...")
    neighborhood_metrics = analyze_neighborhood_structure(G, classifications)
    
    print("\nAnalyzing extended neighborhood structure...")
    extended_metrics = analyze_extended_neighborhood(G, classifications, max_hops=2)
    
    print("\nGenerating structural analysis report...")
    report_path = generate_structural_analysis_report(metrics, classifications, 
                                                     neighborhood_metrics, extended_metrics)
    
    print("\nGenerating centrality visualizations...")
    viz_path = visualize_centrality_distributions(metrics, classifications)
    
    print("\nGenerating network structure visualization...")
    network_viz_path = visualize_network_structure(G, classifications)
    
    print("\nAnalyzing logical strength...")
    strength_report_path = analyze_logical_strength(G, classifications)
    
    print("\nGenerating enhanced report...")
    enhanced_report_path = generate_enhanced_report(metrics, classifications, 
                                                  neighborhood_metrics, extended_metrics)
    
    print("\nComputing advanced metrics...")
    advanced_metrics = compute_advanced_metrics(G, classifications)
    
    print("\nAnalyzing independence patterns...")
    patterns = analyze_independence_patterns(G, classifications, metrics, advanced_metrics)
    
    print("\nGenerating independence patterns report...")
    patterns_report_path = generate_independence_patterns_report(patterns)
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  Structural Analysis Report: {report_path}")
    print(f"  Centrality Visualizations: {viz_path}")
    print(f"  Network Structure Visualization: {network_viz_path}")
    print(f"  Logical Strength Analysis: {strength_report_path}")
    print(f"  Enhanced Report: {enhanced_report_path}")
    print(f"  Independence Patterns Report: {patterns_report_path}")


if __name__ == "__main__":
    main()
