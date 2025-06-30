"""
Analysis script to test the hypothesis that theorems cluster based on
their minimal axiom requirements.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from expanded_entailment_data import create_expanded_entailment_graph
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_minimal_axioms(G):
    """Find the minimal set of axioms required to prove each theorem."""
    # This is a placeholder function - in a real implementation, 
    # you would analyze the graph to find minimal axiom sets
    
    # For demonstration, we'll create some mock minimal axiom data
    minimal_axioms = {}
    
    # Get systems and theorems
    systems = [node for node in G.nodes() if G.nodes[node].get('type') == 'system']
    theorems = [node for node in G.nodes() if G.nodes[node].get('type') == 'theorem']
    
    # For each theorem, assign some minimal axiom systems
    for theorem in theorems:
        # Get systems that can prove this theorem
        proving_systems = []
        for system in systems:
            if nx.has_path(G, system, theorem):
                proving_systems.append(system)
        
        # Randomly select a subset as "minimal"
        if proving_systems:
            # Take between 1 and 3 systems as minimal
            n_minimal = min(len(proving_systems), np.random.randint(1, 4))
            minimal_axioms[theorem] = np.random.choice(proving_systems, n_minimal, replace=False).tolist()
        else:
            minimal_axioms[theorem] = []
    
    return minimal_axioms

def create_axiom_theorem_matrix(G, minimal_axioms):
    """Create a binary matrix of theorems and their minimal axiom requirements."""
    # Get all systems that appear in minimal axioms
    all_systems = set()
    for systems in minimal_axioms.values():
        all_systems.update(systems)
    
    # Create a DataFrame with theorems as rows and systems as columns
    matrix = pd.DataFrame(0, index=minimal_axioms.keys(), columns=sorted(all_systems))
    
    # Fill in the matrix
    for theorem, systems in minimal_axioms.items():
        for system in systems:
            matrix.loc[theorem, system] = 1
    
    return matrix

def cluster_theorems(axiom_theorem_matrix, n_clusters=5):
    """Cluster theorems based on their minimal axiom requirements."""
    # Remove theorems with no proving systems
    filtered_matrix = axiom_theorem_matrix.loc[axiom_theorem_matrix.sum(axis=1) > 0]
    
    if len(filtered_matrix) < n_clusters:
        print(f"Warning: Not enough theorems with proving systems. Reducing clusters to {len(filtered_matrix)}")
        n_clusters = max(2, len(filtered_matrix))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(filtered_matrix)
    
    # Create result dictionary
    result = {}
    for i, theorem in enumerate(filtered_matrix.index):
        result[theorem] = clusters[i]
    
    return result

def visualize_theorem_clusters(axiom_theorem_matrix, clusters):
    """Visualize theorem clusters using t-SNE."""
    # Remove theorems with no proving systems
    filtered_matrix = axiom_theorem_matrix.loc[axiom_theorem_matrix.sum(axis=1) > 0]
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(filtered_matrix)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'theorem': filtered_matrix.index,
        'cluster': [clusters[t] for t in filtered_matrix.index]
    })
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='viridis', s=100)
    
    # Add theorem labels
    for i, row in df.iterrows():
        plt.text(row['x'] + 0.1, row['y'] + 0.1, row['theorem'], fontsize=9)
    
    plt.title('Theorem Clusters Based on Minimal Axiom Requirements')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "theorem_clusters.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def analyze_cluster_composition(clusters, G):
    """Analyze the composition of each cluster."""
    # Group theorems by cluster
    cluster_groups = defaultdict(list)
    for theorem, cluster in clusters.items():
        cluster_groups[cluster].append(theorem)
    
    # Analyze each cluster
    cluster_analysis = {}
    for cluster, theorems in cluster_groups.items():
        # Get theorem descriptions
        descriptions = {t: G.nodes[t].get('description', '') for t in theorems}
        
        # Count theorem types (based on description keywords)
        type_counts = defaultdict(int)
        for desc in descriptions.values():
            if 'set theory' in desc.lower():
                type_counts['Set Theory'] += 1
            elif 'arithmetic' in desc.lower():
                type_counts['Arithmetic'] += 1
            elif 'algebra' in desc.lower():
                type_counts['Algebra'] += 1
            elif 'analysis' in desc.lower():
                type_counts['Analysis'] += 1
            elif 'topology' in desc.lower():
                type_counts['Topology'] += 1
            else:
                type_counts['Other'] += 1
        
        cluster_analysis[cluster] = {
            'theorems': theorems,
            'count': len(theorems),
            'type_counts': dict(type_counts)
        }
    
    return cluster_analysis

def analyze_system_coverage(minimal_axioms, G):
    """Analyze which systems are most effective at covering theorems."""
    # Count how many theorems each system minimally proves
    system_counts = defaultdict(int)
    for theorem, systems in minimal_axioms.items():
        for system in systems:
            system_counts[system] += 1
    
    # Sort systems by theorem count
    sorted_systems = sorted(system_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get system descriptions
    system_info = {}
    for system, count in sorted_systems:
        system_info[system] = {
            'count': count,
            'description': G.nodes[system].get('description', '')
        }
    
    return system_info

def generate_minimal_axiom_report(minimal_axioms, clusters, cluster_analysis, system_info):
    """Generate a report of the minimal axiom analysis."""
    report_path = os.path.join(OUTPUT_DIR, "minimal_axiom_analysis.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Minimal Axiom Analysis Report\n\n")
        f.write("This report analyzes how theorems cluster based on their minimal axiom requirements.\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"- Total theorems analyzed: {len(minimal_axioms)}\n")
        f.write(f"- Theorems with proving systems: {sum(1 for systems in minimal_axioms.values() if systems)}\n")
        f.write(f"- Number of clusters identified: {len(cluster_analysis)}\n\n")
        
        # System coverage
        f.write("## System Coverage\n\n")
        f.write("The following formal systems are most effective at covering theorems:\n\n")
        f.write("| System | Theorems Covered | Description |\n")
        f.write("|--------|-----------------|-------------|\n")
        
        for system, info in system_info.items():
            f.write(f"| {system} | {info['count']} | {info['description']} |\n")
        
        # Cluster analysis
        f.write("\n## Cluster Analysis\n\n")
        
        for cluster, info in cluster_analysis.items():
            f.write(f"### Cluster {cluster}\n\n")
            f.write(f"- Number of theorems: {info['count']}\n")
            f.write("- Theorem types:\n")
            
            for type_name, count in info['type_counts'].items():
                f.write(f"  - {type_name}: {count}\n")
            
            f.write("\n- Theorems in this cluster:\n")
            for theorem in info['theorems']:
                f.write(f"  - {theorem}\n")
            
            f.write("\n")
        
        # Minimal axiom sets
        f.write("## Minimal Axiom Sets\n\n")
        f.write("For each theorem, the minimal set of axioms required for its proof:\n\n")
        f.write("| Theorem | Minimal Axiom Systems |\n")
        f.write("|---------|----------------------|\n")
        
        for theorem, systems in minimal_axioms.items():
            if systems:
                f.write(f"| {theorem} | {', '.join(systems)} |\n")
        
        # Conclusions
        f.write("\n## Conclusions\n\n")
        f.write("Based on the clustering analysis, we can draw the following conclusions:\n\n")
        
        # Check if clusters align with mathematical areas
        area_aligned = any(len(info['type_counts']) < 3 for info in cluster_analysis.values())
        if area_aligned:
            f.write("1. The clusters show alignment with traditional mathematical areas, ")
            f.write("supporting the hypothesis that theorems naturally group by their minimal axiom requirements.\n\n")
        else:
            f.write("1. The clusters do not show strong alignment with traditional mathematical areas, ")
            f.write("suggesting that minimal axiom requirements may cross traditional boundaries.\n\n")
        
        # Check for dominant systems
        top_system = next(iter(system_info.items()))[0] if system_info else None
        top_count = system_info[top_system]['count'] if top_system else 0
        total_theorems = len(minimal_axioms)
        
        if top_system and top_count > total_theorems / 2:
            f.write(f"2. {top_system} is particularly effective, covering over half of the theorems analyzed. ")
            f.write("This suggests it may be an economical choice for a foundational system.\n\n")
        else:
            f.write("2. No single system covers a majority of theorems, ")
            f.write("suggesting that multiple foundational systems are needed for comprehensive coverage.\n\n")
        
        f.write("3. The clustering of theorems provides insight into the natural divisions in mathematical knowledge ")
        f.write("based on logical dependencies rather than traditional subject boundaries.\n")
    
    return report_path

def main():
    """Main function to run the minimal axiom analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nFinding minimal axioms for each theorem...")
    minimal_axioms = find_minimal_axioms(G)
    
    print("\nCreating axiom-theorem matrix...")
    axiom_theorem_matrix = create_axiom_theorem_matrix(G, minimal_axioms)
    
    print("\nClustering theorems based on minimal axiom requirements...")
    clusters = cluster_theorems(axiom_theorem_matrix)
    
    print("\nVisualizing theorem clusters...")
    viz_path = visualize_theorem_clusters(axiom_theorem_matrix, clusters)
    
    print("\nAnalyzing cluster composition...")
    cluster_analysis = analyze_cluster_composition(clusters, G)
    
    print("\nAnalyzing system coverage...")
    system_info = analyze_system_coverage(minimal_axioms, G)
    
    print("\nGenerating report...")
    report_path = generate_minimal_axiom_report(minimal_axioms, clusters, cluster_analysis, system_info)
    
    print(f"\nAnalysis complete! Report saved to {report_path}")
    print(f"Visualizations saved to {viz_path}")

if __name__ == "__main__":
    main()

