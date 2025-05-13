from entailment_theory import EntailmentCone, LogicalStatement, EntailmentRelation
import networkx as nx
from itertools import combinations
import numpy as np
from typing import List, Tuple, Dict, Set

class RelationshipDiscovery:
    """Algorithms for discovering new logical relationships in entailment cones."""
    
    def __init__(self, cone: EntailmentCone):
        self.cone = cone
        
    def apply_transitivity_closure(self) -> List[EntailmentRelation]:
        """
        Discover new relationships through transitivity.
        If A entails B and B entails C, then A entails C.
        """
        new_relations = []
        G = self.cone.graph
        
        for a in G.nodes():
            for c in G.nodes():
                if a != c and not G.has_edge(a, c):
                    # Check if there's a path from a to c
                    try:
                        path = nx.shortest_path(G, a, c)
                        if len(path) > 2:  # There are intermediate nodes
                            # Create a new transitive relation
                            source = self.cone.statements[a]
                            target = self.cone.statements[c]
                            relation = EntailmentRelation(
                                source=source,
                                target=target,
                                relation_type="Transitivity",
                                strength=0.8  # Lower strength for derived relations
                            )
                            new_relations.append(relation)
                    except nx.NetworkXNoPath:
                        continue
        
        return new_relations
    
    def detect_potential_independence(self) -> List[Tuple[str, str]]:
        """
        Detect potential independence relationships.
        Two statements might be independent if:
        1. There's no path between them
        2. They belong to comparable formal systems
        3. They have connections to common statements
        """
        G = self.cone.graph
        potential_independence = []
        
        # Get statements grouped by formal system
        systems = {}
        for node, data in G.nodes(data=True):
            system = data.get('formal_system')
            if system:
                if system not in systems:
                    systems[system] = []
                systems[system].append(node)
        
        # For each formal system, check for potential independence
        for system, statements in systems.items():
            for a, b in combinations(statements, 2):
                # Check if there's no direct path between them
                if not nx.has_path(G, a, b) and not nx.has_path(G, b, a):
                    # Check if they have common neighbors
                    a_neighbors = set(G.successors(a)).union(set(G.predecessors(a)))
                    b_neighbors = set(G.successors(b)).union(set(G.predecessors(b)))
                    common_neighbors = a_neighbors.intersection(b_neighbors)
                    
                    if common_neighbors:
                        potential_independence.append((a, b))
        
        return potential_independence
    
    def infer_new_relations_by_analogy(self) -> List[EntailmentRelation]:
        """
        Infer new relations by analogy with existing patterns.
        If A relates to B similar to how C relates to D, and A relates to C,
        then B might relate to D in the same way.
        """
        G = self.cone.graph
        new_relations = []
        
        # Find all pairs of relations with the same type
        relation_patterns = {}
        for u, v, data in G.edges(data=True):
            rel_type = data.get('relation_type')
            if rel_type not in relation_patterns:
                relation_patterns[rel_type] = []
            relation_patterns[rel_type].append((u, v))
        
        # Look for analogical patterns
        for rel_type, pairs in relation_patterns.items():
            for (a, b), (c, d) in combinations(pairs, 2):
                # Check if A relates to C
                if G.has_edge(a, c) or G.has_edge(c, a):
                    # But B doesn't relate to D yet
                    if not G.has_edge(b, d) and not G.has_edge(d, b):
                        # Suggest a new relation by analogy
                        if b in self.cone.statements and d in self.cone.statements:
                            source = self.cone.statements[b]
                            target = self.cone.statements[d]
                            relation = EntailmentRelation(
                                source=source,
                                target=target,
                                relation_type=f"Analogy_{rel_type}",
                                strength=0.5  # Lower strength for analogical relations
                            )
                            new_relations.append(relation)
        
        return new_relations
