"""
Feature extraction for independence prediction.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TheoremFeatures:
    """Features extracted for a theorem."""
    # Graph-based features
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    clustering_coefficient: float
    
    # Neighborhood features
    neighborhood_size: int
    system_predecessors: int
    theorem_predecessors: int
    independent_neighbors: int
    provable_neighbors: int
    
    # Path-based features
    avg_path_length: float
    path_diversity: float
    
    # Relationship features
    proves_count: int
    implies_count: int
    independent_count: int
    related_count: int
    
    # Semantic features
    concept_diversity: float
    axiom_system_count: int
    reference_count: int

class FeatureExtractor:
    """Extract features for independence prediction."""
    
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self._compute_global_metrics()
    
    def _compute_global_metrics(self):
        """Compute global graph metrics once."""
        self.degree_centrality = nx.degree_centrality(self.G)
        self.betweenness_centrality = nx.betweenness_centrality(self.G)
        self.closeness_centrality = nx.closeness_centrality(self.G)
        self.pagerank = nx.pagerank(self.G)
        self.clustering = nx.clustering(self.G)
    
    def _compute_path_metrics(self, node: str) -> Tuple[float, float]:
        """Compute path-based metrics for a node."""
        try:
            # Average path length to other nodes
            path_lengths = []
            for target in self.G.nodes():
                if target != node:
                    try:
                        path_lengths.append(nx.shortest_path_length(self.G, node, target))
                    except nx.NetworkXNoPath:
                        continue
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            
            # Path diversity (number of different paths)
            path_diversity = 0
            for target in self.G.nodes():
                if target != node:
                    try:
                        paths = list(nx.all_simple_paths(self.G, node, target, cutoff=5))
                        path_diversity += len(paths)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            
            return avg_path_length, path_diversity
            
        except Exception as e:
            print(f"Error computing path metrics for {node}: {str(e)}")
            return 0.0, 0.0
    
    def _count_relationship_types(self, node: str) -> Dict[str, int]:
        """Count different types of relationships for a node."""
        counts = defaultdict(int)
        
        # Count outgoing relationships
        for _, _, data in self.G.out_edges(node, data=True):
            rel_type = data.get('relationship_type', 'unknown')
            counts[rel_type] += 1
        
        # Count incoming relationships
        for _, _, data in self.G.in_edges(node, data=True):
            rel_type = data.get('relationship_type', 'unknown')
            counts[rel_type] += 1
        
        return counts
    
    def _compute_neighborhood_metrics(self, node: str) -> Dict[str, int]:
        """Compute neighborhood-based metrics."""
        predecessors = list(self.G.predecessors(node))
        successors = list(self.G.successors(node))
        
        metrics = {
            'neighborhood_size': len(predecessors) + len(successors),
            'system_predecessors': sum(1 for n in predecessors if self.G.nodes[n].get('type') == 'system'),
            'theorem_predecessors': sum(1 for n in predecessors if self.G.nodes[n].get('type') == 'theorem'),
            'independent_neighbors': sum(1 for n in predecessors + successors 
                                      if self.G.nodes[n].get('classification') == 'independent'),
            'provable_neighbors': sum(1 for n in predecessors + successors 
                                    if self.G.nodes[n].get('classification') == 'provable')
        }
        
        return metrics
    
    def _compute_semantic_features(self, node: str, metadata: Dict) -> Tuple[float, int, int]:
        """Compute semantic-based features."""
        # Concept diversity
        concepts = set(metadata.get('concepts', []))
        concept_diversity = len(concepts)
        
        # Count axiom systems
        axiom_systems = set(metadata.get('axiom_systems', []))
        axiom_count = len(axiom_systems)
        
        # Count references
        references = metadata.get('references', [])
        ref_count = len(references)
        
        return concept_diversity, axiom_count, ref_count
    
    def extract_features(self, node: str, metadata: Dict = None) -> TheoremFeatures:
        """
        Extract all features for a theorem node.
        
        Args:
            node: The theorem node to extract features for
            metadata: Optional metadata dictionary for the theorem
            
        Returns:
            TheoremFeatures object containing all extracted features
        """
        # Get pre-computed centrality metrics
        degree_cent = self.degree_centrality.get(node, 0)
        between_cent = self.betweenness_centrality.get(node, 0)
        close_cent = self.closeness_centrality.get(node, 0)
        pr = self.pagerank.get(node, 0)
        cluster_coef = self.clustering.get(node, 0)
        
        # Compute path metrics
        avg_path_length, path_diversity = self._compute_path_metrics(node)
        
        # Get neighborhood metrics
        neighborhood = self._compute_neighborhood_metrics(node)
        
        # Count relationship types
        rel_counts = self._count_relationship_types(node)
        
        # Compute semantic features if metadata provided
        concept_div, axiom_count, ref_count = (
            self._compute_semantic_features(node, metadata)
            if metadata else (0, 0, 0)
        )
        
        return TheoremFeatures(
            # Graph-based features
            degree_centrality=degree_cent,
            betweenness_centrality=between_cent,
            closeness_centrality=close_cent,
            pagerank=pr,
            clustering_coefficient=cluster_coef,
            
            # Neighborhood features
            neighborhood_size=neighborhood['neighborhood_size'],
            system_predecessors=neighborhood['system_predecessors'],
            theorem_predecessors=neighborhood['theorem_predecessors'],
            independent_neighbors=neighborhood['independent_neighbors'],
            provable_neighbors=neighborhood['provable_neighbors'],
            
            # Path-based features
            avg_path_length=avg_path_length,
            path_diversity=path_diversity,
            
            # Relationship features
            proves_count=rel_counts.get('proves', 0),
            implies_count=rel_counts.get('implies', 0),
            independent_count=rel_counts.get('independent', 0),
            related_count=rel_counts.get('related', 0),
            
            # Semantic features
            concept_diversity=concept_div,
            axiom_system_count=axiom_count,
            reference_count=ref_count
        ) 