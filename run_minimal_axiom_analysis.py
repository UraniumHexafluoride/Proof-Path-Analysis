#!/usr/bin/env python3
"""
Run Minimal Axiom Analysis

This script runs the minimal axiom analysis to test Hypothesis 2:
"There exist natural groupings of theorems based on their minimal axiom requirements."
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
from mathlogic.analysis.minimal_axioms import MinimalAxiomAnalyzer
from expanded_entailment_data import create_expanded_entailment_graph

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Run the minimal axiom analysis."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nInitializing minimal axiom analyzer...")
    analyzer = MinimalAxiomAnalyzer(G)
    
    print("\nFinding minimal axioms for each theorem...")
    minimal_axioms = analyzer.find_minimal_axioms()
    print(f"Found minimal axioms for {len(minimal_axioms)} theorems")
    
    print("\nCreating axiom-theorem matrix...")
    matrix = analyzer.create_axiom_theorem_matrix()
    print(f"Matrix shape: {matrix.shape}")
    
    print("\nClustering theorems based on minimal axiom requirements...")
    clusters = analyzer.cluster_theorems()
    if clusters:
        num_clusters = len(set(clusters.values()))
        print(f"Identified {num_clusters} clusters")
    
    print("\nAnalyzing cluster composition...")
    cluster_analysis = analyzer.analyze_cluster_composition()
    
    print("\nGenerating visualizations...")
    viz_path = analyzer.visualize_theorem_clusters(
        os.path.join(OUTPUT_DIR, "theorem_clusters.png"))
    heatmap_path = analyzer.create_heatmap(
        os.path.join(OUTPUT_DIR, "minimal_axioms_heatmap.png"))
    
    print("\nGenerating report...")
    report_path = analyzer.generate_report(
        os.path.join(OUTPUT_DIR, "minimal_axiom_analysis.md"))
    
    print("\nAnalysis complete!")
    print(f"- Report: {report_path}")
    print(f"- Cluster visualization: {viz_path}")
    print(f"- Heatmap: {heatmap_path}")
    
    # Save minimal axioms to CSV for other analyses
    matrix.to_csv(os.path.join(OUTPUT_DIR, "minimal_axioms_matrix.csv"))

if __name__ == "__main__":
    main()
