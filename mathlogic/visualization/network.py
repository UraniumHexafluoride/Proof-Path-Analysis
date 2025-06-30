"""
Network visualization for theorem relationships.
"""

import os
import logging
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for network visualization."""
    node_size_factor: float = 20
    edge_width_factor: float = 2
    min_edge_width: float = 1
    max_edge_width: float = 5
    show_labels: bool = True
    show_weights: bool = True

class NetworkVisualizer:
    """Network visualization using matplotlib."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Optional directory for saving visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.config = VisualizationConfig()
    
    def plot_metrics(self, metrics: Dict[str, Any]):
        """
        Plot various network metrics.
        
        Args:
            metrics: Dictionary of network metrics
        """
        if not self.output_dir:
            logging.warning("No output directory specified for visualizations")
            return
        
        # Plot degree distribution
        plt.figure(figsize=(10, 6))
        degrees = [d['degree_centrality'] for d in metrics.values() if isinstance(d, dict) and 'degree_centrality' in d]
        if degrees:
            sns.histplot(degrees, bins=20)
            plt.title('Degree Distribution')
            plt.xlabel('Degree Centrality')
            plt.ylabel('Count')
            plt.savefig(os.path.join(self.output_dir, 'degree_distribution.png'))
            plt.close()
        
        # Plot centrality comparison
        plt.figure(figsize=(12, 8))
        centrality_types = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']
        centrality_data = {
            ctype: [d[ctype] for d in metrics.values() if isinstance(d, dict) and ctype in d]
            for ctype in centrality_types
        }
        
        if any(centrality_data.values()):
            sns.boxplot(data=centrality_data)
            plt.title('Centrality Metrics Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'centrality_comparison.png'))
            plt.close()
    
    def plot_network(self, G: nx.DiGraph, metrics: Dict[str, Any], config: Optional[VisualizationConfig] = None):
        """
        Plot the network structure.
        
        Args:
            G: NetworkX DiGraph to visualize
            metrics: Dictionary of network metrics
            config: Optional visualization configuration
        """
        if not self.output_dir:
            logging.warning("No output directory specified for visualizations")
            return
        
        if config:
            self.config = config
        
        plt.figure(figsize=(20, 16))  # Increased figure size
        
        # Calculate optimal k value based on number of nodes
        num_nodes = len(G.nodes())
        k_value = max(1.5, min(4.0, 1.0 / np.sqrt(num_nodes)))  # Dynamic k value
        
        # Try multiple layout algorithms for better spacing
        if num_nodes < 50:
            # For smaller graphs, use spring layout with high k value
            pos = nx.spring_layout(G, k=k_value, iterations=500, seed=42)
        elif num_nodes < 200:
            # For medium graphs, try kamada_kawai which often gives better spacing
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, k=k_value, iterations=300, seed=42)
        else:
            # For large graphs, use faster layout with good spacing
            pos = nx.fruchterman_reingold_layout(G, k=k_value, iterations=100)
        
        # Draw nodes
        node_sizes = [
            self.config.node_size_factor * metrics.get(node, {}).get('degree_centrality', 1)
            for node in G.nodes()
        ]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[metrics.get(node, {}).get('pagerank', 0) for node in G.nodes()],
            cmap=plt.cm.viridis,
            alpha=0.7
        )
        
        # Draw edges
        edge_weights = [
            G[u][v].get('weight', 1)
            for u, v in G.edges()
        ]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[self.config.edge_width_factor * w for w in edge_weights],
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            arrowsize=10
        )
        
        if self.config.show_labels:
            # Reduce font size for better readability in dense graphs
            font_size = max(6, min(10, 200 / num_nodes))  # Dynamic font size
            nx.draw_networkx_labels(G, pos, font_size=font_size)
        
        plt.title('Network Structure')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'network_structure.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_community_structure(self, G: nx.DiGraph, communities: Dict[str, int]):
        """
        Plot the network with communities highlighted.
        
        Args:
            G: NetworkX DiGraph to visualize
            communities: Dictionary mapping nodes to community IDs
        """
        if not self.output_dir:
            logging.warning("No output directory specified for visualizations")
            return
        
        plt.figure(figsize=(20, 16))  # Increased figure size
        
        # Calculate optimal k value based on number of nodes
        num_nodes = len(G.nodes())
        k_value = max(1.5, min(4.0, 1.0 / np.sqrt(num_nodes)))  # Dynamic k value
        
        # Try multiple layout algorithms for better spacing
        if num_nodes < 50:
            # For smaller graphs, use spring layout with high k value
            pos = nx.spring_layout(G, k=k_value, iterations=500, seed=42)
        elif num_nodes < 200:
            # For medium graphs, try kamada_kawai which often gives better spacing
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, k=k_value, iterations=300, seed=42)
        else:
            # For large graphs, use faster layout with good spacing
            pos = nx.fruchterman_reingold_layout(G, k=k_value, iterations=100)
        
        # Draw nodes colored by community
        nx.draw_networkx_nodes(
            G, pos,
            node_color=[communities.get(node, 0) for node in G.nodes()],
            cmap=plt.cm.tab20,
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            edge_color='gray'
        )
        
        if self.config.show_labels:
            # Reduce font size for better readability in dense graphs
            font_size = max(6, min(10, 200 / num_nodes))  # Dynamic font size
            nx.draw_networkx_labels(G, pos, font_size=font_size)
        
        plt.title('Community Structure')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'community_structure.png'), dpi=300, bbox_inches='tight')
        plt.close() 