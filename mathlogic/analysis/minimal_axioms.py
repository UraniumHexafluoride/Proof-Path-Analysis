"""
Minimal Axiom Analysis Module

This module implements analysis of minimal axiom requirements for theorems,
supporting Hypothesis 2: "There exist natural groupings of theorems based on
their minimal axiom requirements."
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

class MinimalAxiomAnalyzer:
    """Analyzes minimal axiom requirements for theorems."""
    
    def __init__(self, G: nx.DiGraph):
        """
        Initialize the analyzer with an entailment graph.
        
        Args:
            G: NetworkX DiGraph representing the entailment graph
        """
        self.G = G
        self.minimal_axioms = {}  # Theorem -> List of minimal axiom systems
        self.axiom_theorem_matrix = None  # DataFrame of theorems x axiom systems
        self.clusters = {}  # Theorem -> cluster ID
        self.cluster_analysis = {}  # Cluster ID -> analysis results
        
    def find_minimal_axioms(self) -> Dict[str, List[str]]:
        """
        Find the minimal set of axioms required to prove each theorem.
        
        A minimal axiom set is one where no proper subset can prove the theorem.
        
        Returns:
            Dict mapping theorem IDs to lists of minimal axiom systems
        """
        # Get systems and theorems
        systems = [node for node in self.G.nodes() 
                  if self.G.nodes[node].get('type') == 'system']
        theorems = [node for node in self.G.nodes() 
                   if self.G.nodes[node].get('type') == 'theorem']
        
        minimal_axioms = {}
        
        for theorem in theorems:
            # Find all systems that can prove this theorem
            proving_systems = []
            for system in systems:
                if nx.has_path(self.G, system, theorem):
                    proving_systems.append(system)
            
            # Find minimal systems (those not contained in others)
            minimal_systems = []
            for system1 in proving_systems:
                is_minimal = True
                for system2 in proving_systems:
                    if system1 != system2 and nx.has_path(self.G, system2, system1):
                        is_minimal = False
                        break
                if is_minimal:
                    minimal_systems.append(system1)
            
            minimal_axioms[theorem] = minimal_systems
        
        self.minimal_axioms = minimal_axioms
        return minimal_axioms
    
    def create_axiom_theorem_matrix(self) -> pd.DataFrame:
        """
        Create a binary matrix of theorems and their minimal axiom requirements.
        
        Returns:
            DataFrame with theorems as rows and axiom systems as columns
        """
        if not self.minimal_axioms:
            self.find_minimal_axioms()
        
        # Get all systems that appear in minimal axioms
        all_systems = set()
        for systems in self.minimal_axioms.values():
            all_systems.update(systems)
        
        # Create a DataFrame with theorems as rows and systems as columns
        matrix = pd.DataFrame(0, 
                             index=self.minimal_axioms.keys(), 
                             columns=sorted(all_systems))
        
        # Fill in the matrix
        for theorem, systems in self.minimal_axioms.items():
            for system in systems:
                matrix.loc[theorem, system] = 1
        
        self.axiom_theorem_matrix = matrix
        return matrix
    
    def cluster_theorems(self, n_clusters=None) -> Dict[str, int]:
        """
        Cluster theorems based on their minimal axiom requirements.
        
        Args:
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Dict mapping theorem IDs to cluster IDs
        """
        if self.axiom_theorem_matrix is None:
            self.create_axiom_theorem_matrix()
        
        # Remove theorems with no proving systems
        filtered_matrix = self.axiom_theorem_matrix.loc[
            self.axiom_theorem_matrix.sum(axis=1) > 0]
        
        if len(filtered_matrix) < 2:
            print("Warning: Not enough theorems with proving systems for clustering")
            return {}
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(filtered_matrix)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(filtered_matrix)
        
        # Create result dictionary
        result = {}
        for i, theorem in enumerate(filtered_matrix.index):
            result[theorem] = int(clusters[i])
        
        self.clusters = result
        return result
    
    def _find_optimal_clusters(self, matrix: pd.DataFrame, max_clusters=10) -> int:
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            matrix: DataFrame with theorems as rows and axiom systems as columns
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        if len(matrix) <= 2:
            return min(len(matrix), 2)
        
        max_k = min(max_clusters, len(matrix) - 1)
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(matrix)
            score = silhouette_score(matrix, clusters)
            silhouette_scores.append((k, score))
        
        # Get k with highest silhouette score
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        return optimal_k
    
    def analyze_cluster_composition(self) -> Dict[int, Dict]:
        """
        Analyze the composition of each cluster.
        
        Returns:
            Dict mapping cluster IDs to analysis results
        """
        if not self.clusters:
            self.cluster_theorems()
        
        # Group theorems by cluster
        cluster_groups = defaultdict(list)
        for theorem, cluster in self.clusters.items():
            cluster_groups[cluster].append(theorem)
        
        # Analyze each cluster
        cluster_analysis = {}
        for cluster, theorems in cluster_groups.items():
            # Get theorem domains (if available)
            domains = {}
            for t in theorems:
                if 'domain' in self.G.nodes[t]:
                    domain = self.G.nodes[t]['domain']
                    domains[t] = domain
                else:
                    # Try to infer domain from description
                    desc = self.G.nodes[t].get('description', '').lower()
                    if 'set theory' in desc:
                        domains[t] = 'Set Theory'
                    elif 'arithmetic' in desc or 'number theory' in desc:
                        domains[t] = 'Arithmetic'
                    elif 'algebra' in desc:
                        domains[t] = 'Algebra'
                    elif 'analysis' in desc:
                        domains[t] = 'Analysis'
                    elif 'topology' in desc:
                        domains[t] = 'Topology'
                    elif 'logic' in desc:
                        domains[t] = 'Logic'
                    else:
                        domains[t] = 'Other'
            
            # Count domains
            domain_counts = defaultdict(int)
            for domain in domains.values():
                domain_counts[domain] += 1
            
            # Get common axiom systems for this cluster
            common_axioms = self._find_common_axioms(theorems)
            
            cluster_analysis[cluster] = {
                'theorems': theorems,
                'count': len(theorems),
                'domain_counts': dict(domain_counts),
                'common_axioms': common_axioms
            }
        
        self.cluster_analysis = cluster_analysis
        return cluster_analysis
    
    def _find_common_axioms(self, theorems: List[str]) -> List[str]:
        """
        Find axiom systems common to all theorems in a cluster.
        
        Args:
            theorems: List of theorem IDs
            
        Returns:
            List of axiom systems common to all theorems
        """
        if not theorems:
            return []
        
        # Get axiom systems for each theorem
        theorem_axioms = [set(self.minimal_axioms.get(t, [])) for t in theorems]
        
        # Find intersection of all axiom sets
        if theorem_axioms:
            common_axioms = set.intersection(*theorem_axioms)
            return sorted(common_axioms)
        return []
    
    def visualize_theorem_clusters(self, save_path):
        """
        Visualize theorem clusters using dimensionality reduction.
        
        Args:
            save_path: Path to save the visualization (if None, will display)
            
        Returns:
            Path to saved visualization or None if displayed
        """
        if self.axiom_theorem_matrix is None:
            self.create_axiom_theorem_matrix()
        
        if not self.clusters:
            self.cluster_theorems()
        
        # Remove theorems with no proving systems
        filtered_matrix = self.axiom_theorem_matrix.loc[
            self.axiom_theorem_matrix.sum(axis=1) > 0]
        
        if len(filtered_matrix) < 2:
            print("Warning: Not enough theorems for visualization")
            return None
        
        # Use PCA instead of t-SNE for small datasets
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(filtered_matrix)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'theorem': filtered_matrix.index,
            'cluster': [self.clusters.get(t, -1) for t in filtered_matrix.index]
        })
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='cluster', data=df, 
                       palette='viridis', s=100)
        
        # Add theorem labels
        for i, row in df.iterrows():
            plt.text(row['x'] + 0.1, row['y'] + 0.1, row['theorem'], 
                    fontsize=9)
        
        plt.title('Theorem Clusters Based on Minimal Axiom Requirements (PCA)')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def create_heatmap(self, output_path=None):
        """
        Create a heatmap of theorems and their minimal axiom systems.
        
        Args:
            output_path: Path to save the heatmap (if None, will display)
            
        Returns:
            Path to saved heatmap or None if displayed
        """
        if self.axiom_theorem_matrix is None:
            self.create_axiom_theorem_matrix()
        
        # Sort theorems by cluster if available
        if self.clusters:
            sorted_theorems = sorted(
                self.axiom_theorem_matrix.index,
                key=lambda t: (self.clusters.get(t, -1), t)
            )
            matrix = self.axiom_theorem_matrix.loc[sorted_theorems]
        else:
            matrix = self.axiom_theorem_matrix
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, cmap='Blues', cbar_kws={'label': 'Required'})
        
        plt.title('Minimal Axiom Requirements for Theorems')
        plt.xlabel('Axiom Systems')
        plt.ylabel('Theorems')
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def generate_report(self, output_path=None) -> str:
        """
        Generate a comprehensive report of the minimal axiom analysis.
        
        Args:
            output_path: Path to save the report (if None, will return as string)
            
        Returns:
            Report as string or path to saved report
        """
        if not self.minimal_axioms:
            self.find_minimal_axioms()
        
        if not self.clusters:
            self.cluster_theorems()
        
        if not self.cluster_analysis:
            self.analyze_cluster_composition()
        
        # Generate report
        report = [
            "# Minimal Axiom Analysis Report",
            "",
            "## Overview",
            "",
            f"- Total theorems analyzed: {len(self.minimal_axioms)}",
            f"- Theorems with proving systems: {sum(1 for systems in self.minimal_axioms.values() if systems)}",
            f"- Number of clusters identified: {len(self.cluster_analysis)}",
            "",
            "## Cluster Analysis",
            ""
        ]
        
        # Add cluster details
        for cluster_id, info in self.cluster_analysis.items():
            report.extend([
                f"### Cluster {cluster_id}",
                "",
                f"- Number of theorems: {info['count']}",
                "- Domain distribution:"
            ])
            
            for domain, count in info['domain_counts'].items():
                percentage = (count / info['count']) * 100
                report.append(f"  - {domain}: {count} ({percentage:.1f}%)")
            
            report.extend([
                "",
                "- Common axiom systems:",
                "  - " + ", ".join(info['common_axioms']) if info['common_axioms'] else "  - None",
                "",
                "- Theorems in this cluster:"
            ])
            
            for theorem in info['theorems']:
                desc = self.G.nodes[theorem].get('description', '')
                report.append(f"  - {theorem}: {desc}")
            
            report.append("")
        
        # Add minimal axiom details
        report.extend([
            "## Minimal Axiom Requirements",
            "",
            "| Theorem | Minimal Axiom Systems |",
            "|---------|----------------------|"
        ])
        
        for theorem, systems in self.minimal_axioms.items():
            if systems:
                report.append(f"| {theorem} | {', '.join(systems)} |")
        
        # Add conclusions
        report.extend([
            "",
            "## Conclusions",
            ""
        ])
        
        # Check if clusters align with mathematical domains
        domain_alignment = self._evaluate_domain_alignment()
        if domain_alignment > 0.7:
            report.append("1. The clusters show strong alignment with traditional mathematical domains, "
                         "supporting the hypothesis that theorems naturally group by their minimal axiom requirements.")
        elif domain_alignment > 0.4:
            report.append("1. The clusters show moderate alignment with traditional mathematical domains, "
                         "partially supporting the hypothesis that theorems group by their minimal axiom requirements.")
        else:
            report.append("1. The clusters show weak alignment with traditional mathematical domains, "
                         "suggesting that minimal axiom requirements may cross traditional boundaries.")
        
        report.append("")
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Save or return report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            return output_path
        else:
            return report_text
    
    def _evaluate_domain_alignment(self) -> float:
        """
        Evaluate how well clusters align with mathematical domains.
        
        Returns:
            Alignment score between 0 (no alignment) and 1 (perfect alignment)
        """
        if not self.cluster_analysis:
            return 0.0
        
        total_theorems = sum(info['count'] for info in self.cluster_analysis.values())
        if total_theorems == 0:
            return 0.0
        
        # Count theorems in the dominant domain of each cluster
        aligned_theorems = 0
        for info in self.cluster_analysis.values():
            if info['domain_counts']:
                dominant_domain = max(info['domain_counts'].items(), key=lambda x: x[1])
                aligned_theorems += dominant_domain[1]
        
        return aligned_theorems / total_theorems


