"""
Visualization utilities for mathematical logic graphs.
Provides functions for creating various types of graph visualizations.
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os

class GraphVisualizer:
    """Handles visualization of mathematical logic graphs."""
    
    def __init__(self, output_dir: str = "entailment_output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self):
        """Configure visualization style settings."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_entailment_visualization(
        self,
        G: nx.DiGraph,
        filename: str,
        show_labels: bool = True,
        node_size: Optional[Dict[str, float]] = None,
        highlight_nodes: Optional[list] = None
    ) -> str:
        """
        Create visualization of entailment relationships.
        
        Args:
            G: NetworkX directed graph
            filename: Output filename
            show_labels: Whether to show node labels
            node_size: Optional dict mapping nodes to sizes
            highlight_nodes: Optional list of nodes to highlight
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(16, 12))
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        node_colors = self._get_node_colors(G, highlight_nodes)
        node_sizes = [node_size.get(node, 1000) if node_size else 1000 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos,
                             node_color=node_colors,
                             node_size=node_sizes)
        
        # Draw edges with different styles per relationship
        self._draw_edges_by_type(G, pos)
        
        # Add labels if requested
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add legend
        self._add_legend()
        
        # Save and close
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _get_node_colors(self, G: nx.DiGraph, highlight_nodes: Optional[list] = None) -> list:
        """Get colors for nodes based on type and highlighting."""
        colors = []
        for node in G.nodes():
            if highlight_nodes and node in highlight_nodes:
                colors.append('red')
            elif G.nodes[node].get('type') == 'system':
                colors.append('lightblue')
            elif G.nodes[node].get('type') == 'theorem':
                colors.append('lightgreen')
            else:
                colors.append('gray')
        return colors
    
    def _draw_edges_by_type(self, G: nx.DiGraph, pos: Dict[str, tuple]):
        """Draw edges with different styles based on relationship type."""
        # Group edges by relationship type
        proves_edges = [(u, v) for u, v, d in G.edges(data=True)
                       if d.get('relation_type') == 'Proves']
        independence_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get('relation_type') == 'Independence']
        contains_edges = [(u, v) for u, v, d in G.edges(data=True)
                         if d.get('relation_type') == 'Contains']
        
        # Draw each type with distinct style
        nx.draw_networkx_edges(G, pos, edgelist=proves_edges,
                             edge_color='blue', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=independence_edges,
                             edge_color='red', style='dashed', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=contains_edges,
                             edge_color='green', arrows=True)
    
    def _add_legend(self):
        """Add legend to the visualization."""
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor='lightblue', label='System'),
            Patch(facecolor='lightgreen', label='Theorem'),
            Line2D([0], [0], color='blue', label='Proves'),
            Line2D([0], [0], color='red', linestyle='--', label='Independence'),
            Line2D([0], [0], color='green', label='Contains')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
    
    def create_metric_visualization(
        self,
        metrics: Dict[str, Dict[str, float]],
        filename: str,
        metric_name: str = 'degree_centrality'
    ) -> str:
        """
        Create visualization of graph metrics.
        
        Args:
            metrics: Dictionary of node metrics
            filename: Output filename
            metric_name: Name of metric to visualize
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Extract metric values
        values = [m[metric_name] for m in metrics.values()]
        labels = list(metrics.keys())
        
        # Create bar plot
        sns.barplot(x=labels, y=values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric_name.replace("_", " ").title()} by Node')
        
        # Save and close
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

def create_full_visualization_suite(
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    output_dir: str = "entailment_output"
) -> Dict[str, str]:
    """
    Create a complete suite of visualizations for analysis results.
    
    Args:
        G: NetworkX directed graph
        metrics: Dictionary of node metrics
        output_dir: Output directory
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    visualizer = GraphVisualizer(output_dir)
    
    # Create main graph visualization
    paths = {
        'entailment_graph': visualizer.create_entailment_visualization(
            G, 'entailment_graph.png'
        )
    }
    
    # Create metric visualizations
    metric_types = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for metric in metric_types:
        paths[f'{metric}_plot'] = visualizer.create_metric_visualization(
            metrics, f'{metric}_distribution.png', metric
        )
    
    return paths

