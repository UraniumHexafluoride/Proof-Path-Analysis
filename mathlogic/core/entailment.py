"""
Entailment relationships between logical statements.

This module defines the core classes for representing logical statements
and the entailment relationships between them, as well as the directed
graph structure (EntailmentCone) that represents these relationships.
"""

import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any, Union
import uuid


class LogicalStatement:
    """
    Represents a single logical statement in a mathematical theory.
    
    Each statement has a unique ID, content, and optional metadata like
    type (axiom, theorem, etc.), domain, and complexity.
    """
    
    def __init__(self, content: str, 
                 statement_type: str = "theorem", 
                 domain: Optional[str] = None,
                 complexity: Optional[float] = None,
                 statement_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a logical statement.
        
        Args:
            content: The text content of the statement
            statement_type: Type of statement (axiom, theorem, lemma, etc.)
            domain: Mathematical domain the statement belongs to
            complexity: Numerical measure of the statement's complexity
            statement_id: Unique identifier for the statement (generated if None)
            metadata: Additional metadata as a dictionary
        """
        self.content = content
        self.statement_type = statement_type
        self.domain = domain
        self.complexity = complexity
        self.id = statement_id if statement_id else str(uuid.uuid4())
        self.metadata = metadata if metadata else {}
        
    def __str__(self) -> str:
        return f"{self.statement_type.capitalize()}: {self.content}"
    
    def __repr__(self) -> str:
        return f"LogicalStatement(id='{self.id}', type='{self.statement_type}', content='{self.content}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, LogicalStatement):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def update_metadata(self, **kwargs) -> None:
        """Update statement metadata with provided key-value pairs."""
        self.metadata.update(kwargs)


class EntailmentRelation:
    """
    Represents a logical entailment relationship between two statements.
    
    This captures that one statement logically follows from another,
    with optional metadata about the nature of the entailment.
    """
    
    def __init__(self, 
                 source: LogicalStatement, 
                 target: LogicalStatement,
                 relation_type: str = "direct",
                 proof_length: Optional[int] = None,
                 proof_difficulty: Optional[float] = None,
                 relation_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an entailment relation.
        
        Args:
            source: The logical statement that entails the target
            target: The logical statement that is entailed by the source
            relation_type: Type of entailment (direct, indirect, etc.)
            proof_length: Estimated length/steps of proof (if known)
            proof_difficulty: Numerical measure of proof difficulty
            relation_id: Unique identifier for the relation (generated if None)
            metadata: Additional metadata as a dictionary
        """
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.proof_length = proof_length
        self.proof_difficulty = proof_difficulty
        self.id = relation_id if relation_id else str(uuid.uuid4())
        self.metadata = metadata if metadata else {}
    
    def __str__(self) -> str:
        return f"{self.source.content} ⊢ {self.target.content}"
    
    def __repr__(self) -> str:
        return f"EntailmentRelation(source='{self.source.id}', target='{self.target.id}', type='{self.relation_type}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, EntailmentRelation):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def update_metadata(self, **kwargs) -> None:
        """Update relation metadata with provided key-value pairs."""
        self.metadata.update(kwargs)


class EntailmentCone:
    """
    Represents a directed graph of logical statements and their entailment relationships.
    
    This is the main data structure for analyzing the structure of a mathematical theory,
    using NetworkX as the underlying graph implementation.
    """
    
    def __init__(self, name: str = "Entailment Cone"):
        """
        Initialize an entailment cone.
        
        Args:
            name: A descriptive name for this entailment cone
        """
        self.name = name
        self.graph = nx.DiGraph(name=name)
        self._statements = {}  # Map from statement ID to LogicalStatement
        self._relations = {}   # Map from relation ID to EntailmentRelation
    
    @property
    def statements(self) -> Dict[str, LogicalStatement]:
        """Get all statements in the cone."""
        return self._statements
    
    @property
    def relations(self) -> Dict[str, EntailmentRelation]:
        """Get all entailment relations in the cone."""
        return self._relations
    
    def add_statement(self, statement: LogicalStatement) -> LogicalStatement:
        """
        Add a logical statement to the cone.
        
        Args:
            statement: The logical statement to add
            
        Returns:
            The added statement
        """
        if statement.id not in self._statements:
            self._statements[statement.id] = statement
            self.graph.add_node(statement.id, 
                               content=statement.content,
                               type=statement.statement_type,
                               domain=statement.domain,
                               complexity=statement.complexity,
                               metadata=statement.metadata)
        return statement
    
    def add_relation(self, relation: EntailmentRelation) -> EntailmentRelation:
        """
        Add an entailment relation to the cone.
        
        Args:
            relation: The entailment relation to add
            
        Returns:
            The added relation
        """
        # Ensure both statements are in the graph
        self.add_statement(relation.source)
        self.add_statement(relation.target)
        
        # Add the relation
        if relation.id not in self._relations:
            self._relations[relation.id] = relation
            self.graph.add_edge(relation.source.id, relation.target.id,
                               relation_id=relation.id,
                               type=relation.relation_type,
                               proof_length=relation.proof_length,
                               proof_difficulty=relation.proof_difficulty,
                               metadata=relation.metadata)
        return relation
    
    def get_statement(self, statement_id: str) -> Optional[LogicalStatement]:
        """Get a statement by its ID."""
        return self._statements.get(statement_id)
    
    def get_relation(self, relation_id: str) -> Optional[EntailmentRelation]:
        """Get a relation by its ID."""
        return self._relations.get(relation_id)
    
    def get_relations_between(self, source_id: str, target_id: str) -> List[EntailmentRelation]:
        """Get all relations between two statements."""
        if not self.graph.has_edge(source_id, target_id):
            return []
        
        edge_data = self.graph.get_edge_data(source_id, target_id)
        if isinstance(edge_data, dict) and 'relation_id' in edge_data:
            # Single relation case
            return [self._relations.get(edge_data['relation_id'])]
        else:
            # Multiple relations case (multigraph)
            return [self._relations.get(data['relation_id']) for data in edge_data.values()
                    if 'relation_id' in data]
    
    def get_entailed_by(self, statement_id: str) -> List[LogicalStatement]:
        """Get all statements entailed by the given statement."""
        if statement_id not in self.graph:
            return []
        return [self._statements[target_id] for target_id in self.graph.successors(statement_id)]
    
    def get_entails(self, statement_id: str) -> List[LogicalStatement]:
        """Get all statements that entail the given statement."""
        if statement_id not in self.graph:
            return []
        return [self._statements[source_id] for source_id in self.graph.predecessors(statement_id)]
    
    def compute_transitive_closure(self) -> nx.DiGraph:
        """Compute the transitive closure of the entailment graph."""
        return nx.transitive_closure(self.graph)
    
    def get_all_paths(self, source_id: str, target_id: str) -> List[List[LogicalStatement]]:
        """Get all paths from source to target statement."""
        if source_id not in self.graph or target_id not in self.graph:
            return []
        
        all_paths = []
        for path in nx.all_simple_paths(self.graph, source_id, target_id):
            all_paths.append([self._statements[node_id] for node_id in path])
        return all_paths
    
    def is_independent(self, statement_id: str) -> bool:
        """Check if a statement is independent (no incoming edges)."""
        return self.graph.in_degree(statement_id) == 0 if statement_id in self.graph else False
    
    def is_derived(self, statement_id: str) -> bool:
        """Check if a statement is derived (has incoming edges)."""
        return self.graph.in_degree(statement_id) > 0 if statement_id in self.graph else False
    
    def get_axioms(self) -> List[LogicalStatement]:
        """Get all axioms (statements with no incoming edges)."""
        return [self._statements[node_id] for node_id in self.graph.nodes 
                if self.graph.in_degree(node_id) == 0 and 
                self._statements[node_id].statement_type.lower() == 'axiom']
    
    def get_theorems(self) -> List[LogicalStatement]:
        """Get all theorems (non-axiom statements)."""
        return [self._statements[node_id] for node_id in self.graph.nodes 
                if self._statements[node_id].statement_type.lower() == 'theorem']
    
    def remove_statement(self, statement_id: str) -> None:
        """Remove a statement and all its relations from the cone."""
        if statement_id in self._statements:
            # First, remove all relations involving this statement
            relations_to_remove = []
            for relation_id, relation in self._relations.items():
                if relation.source.id == statement_id or relation.target.id == statement_id:
                    relations_to_remove.append(relation_id)
            
            for rel_id in relations_to_remove:
                del self._relations[rel_id]
            
            # Then remove the statement itself
            del self._statements[statement_id]
            self.graph.remove_node(statement_id)
    
    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation from the cone."""
        if relation_id in self._relations:
            relation = self._relations[relation_id]
            self.graph.remove_edge(relation.source.id, relation.target.id)
            del self._relations[relation_id]
    
    def __str__(self) -> str:
        return f"EntailmentCone '{self.name}' with {len(self._statements)} statements and {len(self._relations)} relations"
    
    def __repr__(self) -> str:
        return f"EntailmentCone(name='{self.name}', statements={len(self._statements)}, relations={len(self._relations)})"


