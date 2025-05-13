import networkx as nx
from typing import Set, Dict, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class LogicalStatement:
    """Represents a formal logical statement."""
    symbol: str
    description: str
    formal_system: str = None
    is_axiom: bool = False
    
@dataclass
class EntailmentRelation:
    """Represents a logical entailment between statements."""
    source: LogicalStatement
    target: LogicalStatement
    relation_type: str
    strength: float = 1.0  # Quantitative measure of entailment strength
    
class EntailmentCone:
    """
    Formal representation of an entailment cone in mathematical logic.
    
    An entailment cone is a directed graph where:
    - Nodes are logical statements
    - Edges represent entailment relations
    - The structure satisfies specific closure properties
    """
    def __init__(self):
        self.statements: Dict[str, LogicalStatement] = {}
        self.relations: List[EntailmentRelation] = []
        self.graph = nx.DiGraph()
        
    def add_statement(self, statement: LogicalStatement) -> None:
        """Add a logical statement to the entailment cone."""
        self.statements[statement.symbol] = statement
        self.graph.add_node(statement.symbol, 
                           description=statement.description,
                           formal_system=statement.formal_system,
                           is_axiom=statement.is_axiom)
    
    def add_relation(self, relation: EntailmentRelation) -> None:
        """Add an entailment relation to the cone."""
        self.relations.append(relation)
        self.graph.add_edge(relation.source.symbol, 
                           relation.target.symbol,
                           relation_type=relation.relation_type,
                           strength=relation.strength)
    
    def check_closure_properties(self) -> Dict[str, bool]:
        """
        Check if the entailment cone satisfies closure properties:
        1. Transitivity: If A entails B and B entails C, then A entails C
        2. Reflexivity: Every statement entails itself
        3. Consistency: No contradictions in the entailment structure
        """
        properties = {
            "transitivity": True,
            "reflexivity": True,
            "consistency": True
        }
        
        # Check transitivity
        for a in self.graph.nodes():
            for b in self.graph.successors(a):
                for c in self.graph.successors(b):
                    if not self.graph.has_edge(a, c):
                        properties["transitivity"] = False
        
        # Check reflexivity
        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                properties["reflexivity"] = False
        
        # Basic consistency check (no cycles in independence relations)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            independence_cycles = []
            for cycle in cycles:
                if len(cycle) > 1:  # Ignore self-loops
                    has_independence = any(
                        self.graph.edges[cycle[i], cycle[i+1]].get('relation_type') == 'Independence'
                        for i in range(len(cycle)-1)
                    )
                    if has_independence:
                        independence_cycles.append(cycle)
            
            if independence_cycles:
                properties["consistency"] = False
        except:
            # If cycle detection fails, assume consistency is unknown
            properties["consistency"] = None
            
        return properties
    
    def compute_logical_distance(self, statement1: str, statement2: str) -> float:
        """
        Compute the logical distance between two statements.
        Distance is defined as the minimum number of entailment steps needed.
        """
        try:
            path = nx.shortest_path(self.graph, statement1, statement2)
            return len(path) - 1
        except nx.NetworkXNoPath:
            return float('inf')  # No path exists