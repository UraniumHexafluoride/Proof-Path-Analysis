import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any
import concurrent.futures
from functools import partial
import threading
import queue
import glob
import json
from datetime import datetime
import shutil
import re
from mathlogic.utils.graph_cleaning import (
    remove_transitive_edges, normalize_graph_nodes, graph_diagnostics,
    normalize_edge_confidence, tag_edge_source
)

# Output directory for reports and figures
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread-safe print queue
print_queue = queue.Queue()
def safe_print(msg):
    print_queue.put(msg)

def print_worker():
    while True:
        msg = print_queue.get()
        if msg is None:
            break
        print(msg)
        print_queue.task_done()


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
        'Axiom of Choice', 'Swinerton-Dyer Conjecture', 'P vs NP',
        # Add intermediate theorems
        'Zorn\'s Lemma', 'Well-Ordering Theorem', 'Compactness Theorem',
        'Löwenheim–Skolem Theorem', 'Completeness Theorem', 'Incompleteness Theorem',
        'Halting Problem', 'Church-Turing Thesis', 'Recursion Theorem',
        'Fundamental Theorem of Arithmetic', 'Prime Number Theorem'
    ]
    for theorem in theorems:
        G.add_node(theorem, type='theorem')

    # Add edges representing "Proves" relationships
    proves_edges = [
        ('ZFC', "Fermat's Last Theorem"),
        ('PA2', 'Four Color Theorem'),
        ('ZFC', 'Poincaré Conjecture'),
        ('ZFC', 'Hodge Conjecture'),
        ('ZFC+MM', 'Swinerton-Dyer Conjecture'),
        # Add intermediate proof relationships
        ('ZFC', 'Zorn\'s Lemma'),
        ('ZFC', 'Well-Ordering Theorem'),
        ('ZFC', 'Compactness Theorem'),
        ('PA', 'Fundamental Theorem of Arithmetic'),
        ('ZFC', 'Prime Number Theorem'),
        ('PA', 'Completeness Theorem'),
        ('PA2', 'Incompleteness Theorem')
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
        ('PVS+NP', 'P vs NP'),
        # Add intermediate independence relationships
        ('PA', 'Halting Problem'),
        ('PA', 'Church-Turing Thesis'),
        ('ZF', 'Well-Ordering Theorem')
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
        
    # Add theorem-to-theorem implications (creates paths through the graph)
    implies_edges = [
        ('Zorn\'s Lemma', 'Well-Ordering Theorem'),
        ('Well-Ordering Theorem', 'Axiom of Choice'),
        ('Axiom of Choice', 'Zorn\'s Lemma'),
        ('Incompleteness Theorem', "Gödel's Incompleteness"),
        ('Halting Problem', 'P vs NP'),
        ('Prime Number Theorem', 'Riemann Hypothesis'),
        ('Prime Number Theorem', 'Twin Prime Conjecture'),
        ('Fundamental Theorem of Arithmetic', 'Prime Number Theorem'),
        ('Completeness Theorem', 'Compactness Theorem'),
        ('Compactness Theorem', 'Löwenheim–Skolem Theorem')
    ]
    for source, target in implies_edges:
        G.add_edge(source, target, relation='Implies')

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
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
    
    print("Metrics keys:", list(metrics.keys()))
    if metrics:
        print("Sample metric:", next(iter(metrics.values())))
    
    return metrics


def classify_statements(G):
    """
    Classify statements into 'provable', 'independent', or 'unknown'.

    - 'provable': If a theorem is a successor of a system via a 'Proves' edge.
    - 'independent': If a theorem is a successor of a system via an 'Independence' edge.
    - 'related_provable': A successor of a provable theorem.
    - 'related_unknown': Everything else.
    """
    classifications = {}
    provable_nodes = set()
    independent_nodes = set()

    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        
        is_provable = any(
            G.nodes[p].get('type') == 'system' and G[p][node].get('relation') == 'Proves'
            for p in predecessors
        )
        
        is_independent = any(
            G.nodes[p].get('type') == 'system' and G[p][node].get('relation') == 'Independence'
            for p in predecessors
        )

        if is_provable:
            classifications[node] = 'provable'
            provable_nodes.add(node)
        elif is_independent:
            classifications[node] = 'independent'
            independent_nodes.add(node)

    # Second pass to classify successors of provable nodes
    for node in G.nodes():
        if node not in classifications:
            predecessors = list(G.predecessors(node))
            if any(p in provable_nodes for p in predecessors):
                classifications[node] = 'related_provable'
            else:
                classifications[node] = 'related_unknown'
    
    return classifications


def analyze_neighborhood_structure(G, classifications):
    """
    Analyze the neighborhood of each theorem node.
    
    Computes the number of system/theorem predecessors/successors and the
    number of independent/provable neighbors for each theorem.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Dictionary of node classifications.
        
    Returns:
        neighborhood_metrics (dict): Dictionary of neighborhood metrics keyed by node.
    """
    neighborhood_metrics = {}
    
    for node in G.nodes():
        if G.nodes[node].get('type') != 'theorem':
            continue
            
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        
        pred_systems = sum(1 for n in predecessors if G.nodes[n].get('type') == 'system')
        pred_theorems = len(predecessors) - pred_systems
        
        succ_systems = sum(1 for n in successors if G.nodes[n].get('type') == 'system')
        succ_theorems = len(successors) - succ_systems
        
        neighbors = predecessors + successors
        independent_neighbors = sum(1 for n in neighbors if classifications.get(n) == 'independent')
        provable_neighbors = sum(1 for n in neighbors if classifications.get(n) == 'provable')
        
        neighborhood_metrics[node] = {
            'pred_systems': pred_systems,
            'pred_theorems': pred_theorems,
            'succ_systems': succ_systems,
            'succ_theorems': succ_theorems,
            'independent_neighbors': independent_neighbors,
            'provable_neighbors': provable_neighbors,
            'neighborhood_size': len(neighbors)
        }
        
    return neighborhood_metrics


def analyze_extended_neighborhood(G, classifications, max_hops=2):
    """
    Analyze the multi-hop neighborhood around each theorem.
    
    Args:
        G (nx.DiGraph): The entailment graph.
        classifications (dict): Node classifications.
        max_hops (int): The maximum number of hops to explore.
        
    Returns:
        extended_metrics (dict): Dictionary of extended neighborhood metrics.
    """
    extended_metrics = defaultdict(lambda: defaultdict(lambda: {'systems': 0, 'theorems': 0, 'independent': 0, 'provable': 0}))

    for start_node in G.nodes():
        if G.nodes[start_node].get('type') != 'theorem':
            continue
            
        for hop in range(1, max_hops + 1):
            # Explore nodes at exactly `hop` distance
            nodes_at_hop = nx.single_source_shortest_path_length(G, start_node, cutoff=hop)
            
            # Filter to include only nodes at the specified hop
            nodes_at_hop = {node for node, dist in nodes_at_hop.items() if dist == hop}

            for node in nodes_at_hop:
                node_type = G.nodes[node].get('type')
                node_class = classifications.get(node)
                
                if node_type == 'system':
                    extended_metrics[start_node][hop]['systems'] += 1
                elif node_type == 'theorem':
                    extended_metrics[start_node][hop]['theorems'] += 1
                    
                if node_class == 'independent':
                    extended_metrics[start_node][hop]['independent'] += 1
                elif node_class == 'provable':
                    extended_metrics[start_node][hop]['provable'] += 1
                    
    return extended_metrics


def generate_structural_analysis_report(metrics, classifications, neighborhood_metrics, extended_metrics=None, output_dir="."):
    """Generate a markdown report summarizing the structural analysis."""
    
    # Convert metrics to a pandas DataFrame for easier analysis
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['classification'] = df.index.map(classifications)
    
    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Summary of classifications
    classification_counts = df['classification'].value_counts()
    
    # Average metrics by classification
    avg_metrics = df.groupby('classification')[numeric_cols].mean()
    
    # Neighborhood metrics DataFrame
    neighborhood_df = pd.DataFrame.from_dict(neighborhood_metrics, orient='index')
    neighborhood_df['classification'] = neighborhood_df.index.map(classifications)
    numeric_neighborhood_cols = neighborhood_df.select_dtypes(include=np.number).columns.tolist()
    avg_neighborhood = neighborhood_df.groupby('classification')[numeric_neighborhood_cols].mean()

    # Generate the report
    report = "# Structural Analysis of Independence in Mathematics\n\n"
    report += "This report analyzes the structural position of independent statements in the entailment graph.\n\n"
    
    report += "## Classification Summary\n\n"
    report += classification_counts.to_markdown() + "\n\n"
    
    report += "## Centrality Metrics by Classification\n\n"
    report += "Average centrality metrics for each classification:\n\n"
    report += avg_metrics[['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']].to_markdown(floatfmt=".4f") + "\n\n"
    
    report += "## Neighborhood Structure Analysis\n\n"
    report += "Average neighborhood metrics for each classification:\n\n"
    report += avg_neighborhood.to_markdown(floatfmt=".2f") + "\n\n"

    # Extended neighborhood analysis
    if extended_metrics:
        extended_df_data = []
        for node, hops_data in extended_metrics.items():
            for hop, data in hops_data.items():
                extended_df_data.append({
                    'node': node,
                    'classification': classifications.get(node),
                    'hop': hop,
                    'systems': data['systems'],
                    'theorems': data['theorems'],
                    'independent': data['independent'],
                    'provable': data['provable']
                })
        extended_df = pd.DataFrame(extended_df_data)

        if not extended_df.empty:
            # Exclude 'hop' from the columns to be aggregated, since it's an index
            numeric_extended_cols = extended_df.select_dtypes(include=np.number).columns.tolist()
            if 'hop' in numeric_extended_cols:
                numeric_extended_cols.remove('hop')
            
            avg_extended = extended_df.groupby(['classification', 'hop'])[numeric_extended_cols].mean().reset_index()
            
            # Now, safely calculate the total
            if 'systems' in avg_extended and 'theorems' in avg_extended:
                 avg_extended['total'] = avg_extended['systems'] + avg_extended['theorems']
            
            report += "## Extended Neighborhood Analysis\n\n"
            report += "This section analyzes the multi-hop neighborhood structure around theorems.\n\n"
            report += "### Average Neighborhood Composition by Distance\n\n"
            report += avg_extended.to_markdown(index=False, floatfmt=".2f") + "\n\n"

            # Shannon Diversity
            diversity_data = []
            for node, n_metrics in neighborhood_metrics.items():
                counts = [
                    n_metrics['pred_systems'], n_metrics['pred_theorems'],
                    n_metrics['succ_systems'], n_metrics['succ_theorems']
                ]
                counts = [c for c in counts if c > 0]
                if sum(counts) > 0:
                    probs = np.array(counts) / sum(counts)
                    shannon_diversity = -np.sum(probs * np.log2(probs))
                    diversity_data.append({'node': node, 'shannon_diversity': shannon_diversity})
            
            if diversity_data:
                diversity_df = pd.DataFrame(diversity_data)
                diversity_df['classification'] = diversity_df['node'].map(classifications)
                avg_diversity = diversity_df.groupby('classification')['shannon_diversity'].mean()
                
                report += "### Neighborhood Diversity Analysis\n\n"
                report += "Shannon diversity index measures the diversity of node types in the neighborhood.\n"
                report += "Higher values indicate more diverse neighborhoods.\n\n"
                report += avg_diversity.to_frame().to_markdown(floatfmt=".4f") + "\n\n"

    report += "## Key Findings\n\n"
    report += "- The comparison of betweenness suggests a difference in how central independent or provable statements are.\n"
    report += "- Degree and PageRank differences might indicate varying levels of influence within the network.\n"
    report += "- Neighborhood metrics reveal potential clustering of certain classifications.\n\n"
    
    report += "## Visualizations\n\n"
    report += "The following figures have been generated in this run's directory.\n\n"
    
    # Save the report
    report_path = os.path.join(output_dir, "structural_independence_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    safe_print(f"Structural analysis report saved to: {report_path}")
    return report_path


def visualize_centrality_distributions(metrics, classifications, output_dir="."):
    """Visualize distributions of centrality metrics for each classification."""
    
    if not metrics:
        safe_print("No metrics to visualize.")
        return

    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['classification'] = df.index.map(classifications)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Centrality Metric Distributions by Classification', fontsize=16)
    
    # Plot distributions
    sns.boxplot(x='classification', y='degree_centrality', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Degree Centrality')
    
    sns.boxplot(x='classification', y='betweenness_centrality', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Betweenness Centrality')
    
    sns.boxplot(x='classification', y='closeness_centrality', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Closeness Centrality')
    
    sns.boxplot(x='classification', y='pagerank', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('PageRank')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    viz_path = os.path.join(output_dir, "centrality_distributions.png")
    plt.savefig(viz_path)
    plt.close()
    
    safe_print(f"Centrality distributions visualization saved to: {viz_path}")
    return viz_path


def visualize_network_structure(G, classifications, output_dir="."):
    """Visualize the overall network structure."""
    
    if G.number_of_nodes() == 0:
        safe_print("Graph is empty, cannot visualize.")
        return

    plt.figure(figsize=(20, 20))
    
    # Node colors based on classification
    color_map = {
        'provable': '#4CAF50',        # Green
        'independent': '#F44336',     # Red
        'related_provable': '#81C784', # Light Green
        'related_unknown': '#BDBDBD',  # Grey
        'system': '#2196F3'           # Blue
    }
    
    node_colors = [color_map.get(classifications.get(node, 'related_unknown'), '#BDBDBD') for node in G.nodes()]
    node_sizes = [1000 if G.nodes[node].get('type') == 'system' else 300 for node in G.nodes()]
    
    # Use a spring layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Draw labels for systems and important theorems only
    labels = {
        node: node for node in G.nodes() 
        if G.nodes[node].get('type') == 'system' or G.degree(node) > 10
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Entailment Graph Network Structure", fontsize=20)
    plt.axis('off')
    
    # Save figure
    network_viz_path = os.path.join(output_dir, "network_structure.png")
    plt.savefig(network_viz_path, bbox_inches='tight')
    plt.close()
    
    safe_print(f"Network structure visualization saved to: {network_viz_path}")
    return network_viz_path


def analyze_logical_strength(G, classifications, output_dir="."):
    """Analyze the logical strength and dependencies of theorems."""
    
    # Calculate logical strength as a weighted combination of in-degree and PageRank
    strength_metrics = {}
    pagerank = nx.pagerank(G)
    
    for node in G.nodes():
        if G.nodes[node].get('type') == 'theorem':
            in_degree = G.in_degree(node)
            strength = (0.4 * in_degree) + (0.6 * pagerank.get(node, 0) * 100)
            strength_metrics[node] = strength
    
    df = pd.DataFrame.from_dict(strength_metrics, orient='index', columns=['logical_strength'])
    df['classification'] = df.index.map(classifications)
    
    report = "# Logical Strength Analysis\n\n"
    report += "This report analyzes the logical strength of theorems based on their position in the entailment graph.\n\n"
    
    report += "## Average Logical Strength by Classification\n\n"
    avg_strength = df.groupby('classification')['logical_strength'].mean()
    report += avg_strength.to_frame().to_markdown(floatfmt=".4f") + "\n\n"
    
    report += "## Top 10 Strongest Theorems\n\n"
    top_10 = df.sort_values('logical_strength', ascending=False).head(10)
    report += top_10.to_markdown(floatfmt=".4f") + "\n\n"
    
    # Save the report
    strength_report_path = os.path.join(output_dir, "logical_strength_analysis.md")
    with open(strength_report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    safe_print(f"Logical strength analysis saved to: {strength_report_path}")
    return strength_report_path


def generate_enhanced_report(metrics, classifications, neighborhood_metrics, extended_metrics=None, output_dir="."):
    """Generate an enhanced report with interpretations."""
    
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['classification'] = df.index.map(classifications)
    
    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    avg_metrics = df.groupby('classification')[numeric_cols].mean()
    
    neighborhood_df = pd.DataFrame.from_dict(neighborhood_metrics, orient='index')
    neighborhood_df['classification'] = neighborhood_df.index.map(classifications)
    numeric_neighborhood_cols = neighborhood_df.select_dtypes(include=np.number).columns.tolist()
    avg_neighborhood = neighborhood_df.groupby('classification')[numeric_neighborhood_cols].mean()
    
    report = "# Enhanced Independence Analysis\n\n"
    report += "This report provides extended interpretations of the structural analysis.\n\n"
    
    report += "## Interpretation of Centrality Metrics\n\n"
    report += "Centrality metrics reveal the structural importance of theorems in the mathematical landscape:\n\n"
    
    for metric in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']:
        if metric in avg_metrics.columns:
            metric_name = metric.replace('_', ' ').title()
            report += f"### {metric_name}\n\n"
            report += f"{metric_name} measures the importance of a theorem in the network.\n\n"
            top_class = avg_metrics[metric].idxmax()
            top_value = avg_metrics[metric].max()
            report += f"- **{top_class}** theorems have the highest average {metric_name.lower()} ({top_value:.4f}), suggesting they are structurally significant.\n\n"

    report += "## Interpretation of Neighborhood Structure\n\n"
    report += "The neighborhood structure reveals how theorems relate to their immediate surroundings:\n\n"
    
    if 'pred_systems' in avg_neighborhood:
        pred_systems_class = avg_neighborhood['pred_systems'].idxmax()
        pred_systems_value = avg_neighborhood['pred_systems'].max()
        report += "### Formal System Dependencies\n\n"
        report += f"- **{pred_systems_class}** theorems are proven by more formal systems ({pred_systems_value:.2f} on average), suggesting they are more fundamental.\n\n"

    # Save the report
    enhanced_report_path = os.path.join(output_dir, "enhanced_independence_analysis.md")
    with open(enhanced_report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    safe_print(f"Enhanced report saved to: {enhanced_report_path}")
    return enhanced_report_path


def load_latest_scraping_output(scraped_data_dir="entailment_output/scraped_data"):
    """Finds the latest scraping output file and returns its path and loaded data."""
    list_of_files = glob.glob(os.path.join(scraped_data_dir, 'improved_scraping_results_*.json'))
    if not list_of_files:
        safe_print("No scraping output files found.")
        return None, None
    
    latest_file = max(list_of_files, key=os.path.getctime)
    safe_print(f"Loading latest scraping output: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return latest_file, data
    except (json.JSONDecodeError, IOError) as e:
        safe_print(f"Error loading {latest_file}: {e}")
        return latest_file, None

def build_graph_from_scraping(data):
    import networkx as nx
    G = nx.DiGraph()
    
    edges_added = 0
    relationships_processed = 0
    
    # Load relationships from the main vertices in entailment results
    if "results" in data:
        # Collect all results from all sources
        all_results = []
        for source_results in data.get("results", {}).values():
            all_results.extend(source_results)

        safe_print(f"Processing {len(all_results)} entries from scraped results...")
        
        # First pass: collect all unique theorem names
        all_theorem_names = set()
        for entry in all_results:
            if not entry:
                continue
            
            # Add the main theorem name
            node_id = entry.get("name") or entry.get("title") or str(entry)
            if node_id and isinstance(node_id, str):
                all_theorem_names.add(node_id.strip())
            
            # Add all theorem names mentioned in relationships
            for rel in entry.get("relationships", []):
                source_theorem = rel.get("source_theorem")
                target_theorem = rel.get("target_theorem")
                
                if source_theorem and isinstance(source_theorem, str):
                    all_theorem_names.add(source_theorem.strip())
                if target_theorem and isinstance(target_theorem, str):
                    all_theorem_names.add(target_theorem.strip())
        
        safe_print(f"Found {len(all_theorem_names)} unique theorem names")
        
        # Add all theorems as nodes
        for theorem_name in all_theorem_names:
            G.add_node(theorem_name, type='theorem')
        
        # Second pass: add all relationships as edges
        for entry in all_results:
            if not entry:
                continue
            
            for rel in entry.get("relationships", []):
                relationships_processed += 1
                source_theorem = rel.get("source_theorem")
                target_theorem = rel.get("target_theorem")
                relationship_type = rel.get("relationship_type", "related")
                confidence = rel.get("confidence", 0.5)

                if source_theorem and target_theorem:
                    source_clean = source_theorem.strip()
                    target_clean = target_theorem.strip()
                    
                    if source_clean and target_clean and source_clean != target_clean:
                        G.add_edge(source_clean, target_clean, 
                                 relation=relationship_type, 
                                 confidence=confidence)
                        edges_added += 1

    # Add formal systems and connect them to theorems with meaningful relationships
    systems = ['ZFC', 'PA', 'ZF', 'ACA0', 'ZFC+LC', 'ZFC+MM', 'ZFC+AD', 'PVS+NP']
    for system in systems:
        G.add_node(system, type='system')

    # Load system-theorem relationships from JSON file
    mapping_file = 'system_theorem_relationships.json'
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            system_theorem_relationships = json.load(f)
    else:
        safe_print(f"Warning: {mapping_file} not found. No manual system-theorem edges will be added.")
        system_theorem_relationships = {}

    # Add system-theorem relationships
    for system, relationships in system_theorem_relationships.items():
        for theorem, relation_type in relationships:
            if theorem in G.nodes():
                G.add_edge(system, theorem, relation=relation_type, confidence=0.8)
                edges_added += 1

    # Add system hierarchy relationships
    system_hierarchy = [
        ('ZFC', 'ZF'),
        ('ZFC', 'PA'),
        ('ZF', 'PA'),
        ('ZFC+LC', 'ZFC'),
        ('ZFC+MM', 'ZFC'),
        ('ZFC+AD', 'ZFC'),
        ('ACA0', 'PA')
    ]
    
    for stronger, weaker in system_hierarchy:
        if stronger in G.nodes() and weaker in G.nodes():
            G.add_edge(stronger, weaker, relation='Contains', confidence=0.9)
            edges_added += 1

    safe_print(f"\nGraph construction summary:")
    safe_print(f"  Relationships processed: {relationships_processed}")
    safe_print(f"  Edges added: {edges_added}")
    safe_print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Diagnostics
    node_types = defaultdict(int)
    for node in G.nodes:
        node_types[G.nodes[node].get('type', 'unknown')] += 1
    
    for n_type, count in node_types.items():
        safe_print(f"  {n_type.title()} nodes: {count}")
    
    # Show some example relationships
    if G.number_of_edges() > 0:
        sample_edges = list(G.edges(data=True))[:10]
        safe_print(f"\nSample relationships:")
        for source, target, data in sample_edges:
            relation = data.get('relation', 'unknown')
            safe_print(f"  {source} --[{relation}]--> {target}")
    
    return G


def parallel_compute_metrics(G, node_batch):
    """Compute metrics for a batch of nodes in parallel"""
    metrics = {}
    for node in node_batch:
        # Compute individual node metrics
        node_type = G.nodes[node].get('type', 'unknown')
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        metrics[node] = {
            'type': node_type,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'degree_ratio': out_degree / in_degree if in_degree > 0 else float('inf')
        }
    return metrics

def compute_structural_metrics_parallel(G, num_threads=4):
    """Parallel version of compute_structural_metrics"""
    # Pre-compute global metrics that can't be parallelized
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    
    try:
        hits = nx.hits(G)
        authority_scores = hits[0]
        hub_scores = hits[1]
    except Exception as e:
        safe_print(f"HITS computation failed: {e}")
        authority_scores = {node: 0.0 for node in G.nodes()}
        hub_scores = {node: 0.0 for node in G.nodes()}
    
    # Split nodes into batches
    nodes = list(G.nodes())
    batch_size = max(1, len(nodes) // num_threads)
    node_batches = [nodes[i:i + batch_size] for i in range(0, len(nodes), batch_size)]
    
    # Process batches in parallel
    metrics = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {
            executor.submit(parallel_compute_metrics, G, batch): batch 
            for batch in node_batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_metrics = future.result()
            metrics.update(batch_metrics)
    
    # Add global metrics to results
    for node in metrics:
        metrics[node].update({
            'degree_centrality': degree_centrality[node],
            'in_degree_centrality': in_degree_centrality[node],
            'out_degree_centrality': out_degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'pagerank': pagerank[node],
            'authority_score': authority_scores[node],
            'hub_score': hub_scores[node]
        })
    
    return metrics

def parallel_analyze_neighborhood(G, node_batch, classifications):
    """Analyze neighborhood for a batch of nodes in parallel"""
    neighborhood_metrics = {}
    for node in node_batch:
        if G.nodes[node].get('type') != 'theorem':
            continue
        
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        
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

def analyze_neighborhood_structure_parallel(G, classifications, num_threads=4):
    """Parallel version of analyze_neighborhood_structure"""
    # Split nodes into batches
    nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'theorem']
    batch_size = max(1, len(nodes) // num_threads)
    node_batches = [nodes[i:i + batch_size] for i in range(0, len(nodes), batch_size)]
    
    # Process batches in parallel
    neighborhood_metrics = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {
            executor.submit(parallel_analyze_neighborhood, G, batch, classifications): batch 
            for batch in node_batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_metrics = future.result()
            neighborhood_metrics.update(batch_metrics)
    
    return neighborhood_metrics

def analyze_proof_paths(G, classifications, output_dir="."):
    """Finds and analyzes proof paths in the entailment graph."""
    safe_print("\nAnalyzing proof paths...")
    
    provable_theorems = {node for node, data in G.nodes(data=True) if classifications.get(node) == 'provable'}
    systems = {node for node, data in G.nodes(data=True) if data.get('type') == 'system'}
    
    report = "# Proof Path Analysis\n\n"
    report += "This section details the shortest proof paths found from formal systems to major provable theorems.\n\n"
    
    path_found = False
    for theorem in sorted(list(provable_theorems)):
        for system in sorted(list(systems)):
            if G.has_node(system) and G.has_node(theorem):
                try:
                    # Find all simple paths and take the shortest one
                    paths = list(nx.all_simple_paths(G, source=system, target=theorem, cutoff=10))
                    if paths:
                        shortest_path = min(paths, key=len)
                        path_found = True
                        report += f"## Path to: {theorem}\n\n"
                        report += f"**From System:** {system}\n"
                        report += f"**Path:** `{' -> '.join(shortest_path)}`\n"
                        report += f"**Length:** {len(shortest_path) - 1}\n\n"
                        
                        # Only report one shortest path per theorem
                        break
                except nx.NetworkXNoPath:
                    continue
    
    if not path_found:
        report += "No direct proof paths were found in the graph with the current structure.\n"

    # Save the report
    path_report_path = os.path.join(output_dir, "proof_path_analysis.md")
    with open(path_report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    safe_print(f"Proof path analysis saved to: {path_report_path}")
    return path_report_path

def visualize_entailment_cone(G, theorem_name, classifications, output_dir="."):
    """Visualizes the entailment cone for a specific theorem."""
    if not G.has_node(theorem_name):
        safe_print(f"Cannot generate cone for '{theorem_name}' as it is not in the graph.")
        return None

    safe_print(f"Generating entailment cone for: {theorem_name}")
    
    # The entailment cone is the set of all ancestors of the theorem
    ancestors = nx.ancestors(G, theorem_name)
    ancestors.add(theorem_name) # Include the theorem itself
    
    # Create the subgraph
    cone_graph = G.subgraph(ancestors)
    
    if cone_graph.number_of_nodes() == 0:
        safe_print(f"Entailment cone for '{theorem_name}' is empty.")
        return None

    plt.figure(figsize=(15, 10))
    
    # Node colors based on classification
    color_map = {
        'provable': '#4CAF50', 'independent': '#F44336', 'related_provable': '#81C784',
        'related_unknown': '#BDBDBD', 'system': '#2196F3'
    }
    node_colors = [color_map.get(classifications.get(node, 'related_unknown'), '#BDBDBD') for node in cone_graph.nodes()]
    node_sizes = [800 if G.nodes[node].get('type') == 'system' else 250 for node in cone_graph.nodes()]

    # Use a layout that emphasizes hierarchy
    try:
        pos = nx.nx_agraph.graphviz_layout(cone_graph, prog='dot')
    except:
        safe_print("Graphviz not found, falling back to spring layout for cone.")
        pos = nx.spring_layout(cone_graph)

    nx.draw_networkx_nodes(cone_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(cone_graph, pos, alpha=0.3, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(cone_graph, pos, font_size=8)
    
    plt.title(f"Entailment Cone for {theorem_name}", fontsize=18)
    plt.axis('off')
    
    # Save figure
    cone_viz_path = os.path.join(output_dir, f"entailment_cone_{theorem_name.replace(' ', '_')}.png")
    plt.savefig(cone_viz_path, bbox_inches='tight')
    plt.close()
    
    safe_print(f"Entailment cone visualization saved to: {cone_viz_path}")
    return cone_viz_path

def main():
    """
    Main function with parallel processing support
    """
    # Start print worker thread
    print_thread = threading.Thread(target=print_worker)
    print_thread.start()
    
    try:
        # --- Run Setup ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join("analysis_output", f"run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
        safe_print(f"Created analysis run directory: {run_output_dir}")

        # Load latest scraping output and build graph from it
        safe_print("Loading latest scraping output and building graph...")
        scraped_data_path, data = load_latest_scraping_output()
        
        if not data:
            safe_print("No data to process. Exiting.")
            print_queue.put(None)
            print_thread.join()
            return

        # For reproducibility, copy the source data to the run directory
        if scraped_data_path:
            shutil.copy(scraped_data_path, run_output_dir)
            
        G = build_graph_from_scraping(data)

        # Determine number of threads based on CPU count
        num_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 threads

        safe_print(f"\nComputing structural metrics using {num_threads} threads...")
        metrics = compute_structural_metrics_parallel(G, num_threads)

        safe_print("\nClassifying statements...")
        classifications = classify_statements(G)

        safe_print(f"\nAnalyzing neighborhood structure using {num_threads} threads...")
        neighborhood_metrics = analyze_neighborhood_structure_parallel(G, classifications, num_threads)

        # Run analysis tasks in parallel
        safe_print("\nRunning analysis tasks in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit analysis tasks that don't involve matplotlib
            strength_future = executor.submit(analyze_logical_strength, G, classifications, output_dir=run_output_dir)
            extended_metrics_future = executor.submit(analyze_extended_neighborhood, G, classifications, max_hops=2)
            path_analysis_future = executor.submit(analyze_proof_paths, G, classifications, output_dir=run_output_dir)

            # Wait for analysis tasks to complete
            extended_metrics = extended_metrics_future.result()
            strength_report_path = strength_future.result()
            path_report_path = path_analysis_future.result()

            # Submit report generation tasks
            report_future = executor.submit(
                generate_structural_analysis_report,
                metrics, classifications, neighborhood_metrics, extended_metrics, output_dir=run_output_dir
            )
            enhanced_report_future = executor.submit(
                generate_enhanced_report,
                metrics, classifications, neighborhood_metrics, extended_metrics, output_dir=run_output_dir
            )

            # Get report results
            report_path = report_future.result()
            enhanced_report_path = enhanced_report_future.result()

        # Run visualization tasks in the main thread
        safe_print("\nGenerating visualizations...")
        viz_path = visualize_centrality_distributions(metrics, classifications, output_dir=run_output_dir)
        network_viz_path = visualize_network_structure(G, classifications, output_dir=run_output_dir)
        cone_viz_path = visualize_entailment_cone(G, "Fermat's Last Theorem", classifications, output_dir=run_output_dir)

        safe_print(f"\nAnalysis complete! Results saved to: {run_output_dir}")
        safe_print(f"  - Structural Analysis Report: {os.path.basename(report_path)}")
        safe_print(f"  - Centrality Visualizations: {os.path.basename(viz_path)}")
        safe_print(f"  - Network Structure Visualization: {os.path.basename(network_viz_path)}")
        safe_print(f"  - Logical Strength Analysis: {os.path.basename(strength_report_path)}")
        safe_print(f"  - Enhanced Report: {os.path.basename(enhanced_report_path)}")
        safe_print(f"  - Proof Path Analysis: {os.path.basename(path_report_path)}")
        safe_print(f"  - Entailment Cone for Fermat's Last Theorem generated.")

    finally:
        # Signal print worker to stop and wait for it
        print_queue.put(None)
        print_thread.join()

if __name__ == "__main__":
    main()
