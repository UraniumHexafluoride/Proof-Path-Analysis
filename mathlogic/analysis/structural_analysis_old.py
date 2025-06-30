from entailment_theory import EntailmentCone
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class StructuralAnalysis:
    """Tools for analyzing the structure of entailment cones."""
    
    def __init__(self, cone: EntailmentCone):
        self.cone = cone
        self.G = cone.graph
    
    def identify_bottlenecks(self) -> Dict[str, float]:
        """
        Identify bottleneck statements in the entailment structure.
        Bottlenecks are statements with high betweenness centrality.
        """
        return nx.betweenness_centrality(self.G)
    
    def find_logical_clusters(self) -> Dict[str, Set[str]]:
        """
        Identify clusters of closely related statements.
        Uses community detection algorithms to find logical clusters.
        """
        # Convert to undirected for community detection
        undirected_G = self.G.to_undirected()
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(undirected_G)
            
            # Group nodes by community
            communities = defaultdict(set)
            for node, community_id in partition.items():
                communities[community_id].add(node)
                
            return dict(communities)
        except ImportError:
            # Fallback to connected components if community detection is unavailable
            components = nx.connected_components(undirected_G)
            return {i: set(comp) for i, comp in enumerate(components)}
    
    def analyze_independence_structure(self) -> Dict[str, List[str]]:
        """
        Analyze the structure of independence results.
        Identifies which statements are independent of which formal systems.
        """
        independence_structure = defaultdict(list)
        
        for u, v, data in self.G.edges(data=True):
            if data.get('relation_type') == 'Independence':
                # Get the formal system of the source
                source_system = self.G.nodes[u].get('formal_system')
                if source_system:
                    independence_structure[source_system].append(v)
        
        return dict(independence_structure)
    
    def compute_logical_hierarchy(self) -> Dict[str, int]:
        """
        Compute a hierarchical ranking of statements based on their position
        in the entailment structure.
        """
        # Use PageRank to rank statements by their logical importance
        pagerank = nx.pagerank(self.G)
        
        # Identify source nodes (axioms or base statements)
        sources = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        
        # Compute logical levels based on distance from sources
        levels = {}
        for source in sources:
            # For each node, find shortest path from any source
            for node in self.G.nodes():
                try:
                    path_length = nx.shortest_path_length(self.G, source, node)
                    # Take the minimum level if reachable from multiple sources
                    if node not in levels or path_length < levels[node]:
                        levels[node] = path_length
                except nx.NetworkXNoPath:
                    # If no path exists, don't update the level
                    pass
        
        # For nodes not reachable from any source, assign a high level
        max_level = max(levels.values()) if levels else 0
        for node in self.G.nodes():
            if node not in levels:
                levels[node] = max_level + 1
                
        return levels
    
    def identify_independence_bridges(self) -> List[str]:
        """
        Identify statements that bridge independence results.
        These are statements that can prove theorems independent of weaker systems.
        """
        bridges = []
        independence_edges = [(u, v) for u, v, data in self.G.edges(data=True) 
                             if data.get('relation_type') == 'Independence']
        
        for source, target in independence_edges:
            # Find nodes that can reach the target but not through the source
            for node in self.G.nodes():
                if node != source and node != target:
                    # Check if node can reach target
                    if nx.has_path(self.G, node, target):
                        # Check if source cannot reach target
                        if not nx.has_path(self.G, source, target):
                            bridges.append(node)
        
        return list(set(bridges))  # Remove duplicates
