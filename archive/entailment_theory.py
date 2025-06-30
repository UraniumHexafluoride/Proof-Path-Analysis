import networkx as nx
from typing import Set, Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LogicalStatement:
    """
    Represents a logical statement in mathematical logic.
    
    This can be a theorem, axiom, conjecture, or any other formal statement.
    """
    symbol: str
    description: str = ""
    statement_type: str = "theorem"  # theorem, axiom, conjecture, etc.
    formal_system: str = None  # The formal system this statement belongs to
    is_axiom: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'description': self.description,
            'type': self.statement_type,
            'formal_system': self.formal_system,
            'is_axiom': self.is_axiom,
            'metadata': self.metadata
        }
        
@dataclass
class EntailmentRelation:
    """
    Represents a relationship between two logical statements.
    
    This can be a proof, independence result, or other relationship.
    """
    source: str
    target: str
    relation_type: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'source': self.source,
            'target': self.target,
            'type': self.relation_type,
            'metadata': self.metadata
        }

class EntailmentCone:
    """
    Formal representation of an entailment cone in mathematical logic.
    
    An entailment cone is a directed graph where:
    - Nodes are logical statements (theorems, axioms, conjectures)
    - Edges represent entailment relations (proves, independence, contains)
    - The structure captures the logical dependencies in mathematics
    """
    def __init__(self, name: str = "Entailment Cone"):
        self.name = name
        self.statements = {}  # Dict[str, LogicalStatement]
        self.relations = []  # List[EntailmentRelation]
        self.formal_systems = {}  # Dict[str, Dict]
        self.graph = nx.DiGraph()
        self.metadata = {
            'last_modified': datetime.now(),
            'statement_count': 0,
            'relation_count': 0,
            'validation_status': 'unvalidated'
        }
    
    def add_statement(self, name: str, data: Dict) -> None:
        """Add a logical statement to the cone."""
        if name in self.statements:
            raise ValueError(f"Statement {name} already exists")
            
        statement = LogicalStatement(
            symbol=name,
            description=data.get('description', ''),
            statement_type=data.get('statement_type', 'theorem'),
            formal_system=data.get('formal_system'),
            is_axiom=data.get('is_axiom', False),
            metadata=data
        )
        
        self.statements[name] = statement
        self.graph.add_node(name, **data)
        self.metadata['statement_count'] += 1
        self.metadata['last_modified'] = datetime.now()
    
    def add_formal_system(self, name: str, data: Dict) -> None:
        """Add a formal system to the cone."""
        if name in self.formal_systems:
            raise ValueError(f"Formal system {name} already exists")
            
        self.formal_systems[name] = data
        
        # Add as a special type of statement
        system_data = data.copy()
        system_data['statement_type'] = 'system'
        system_data['is_axiom'] = True
        self.add_statement(name, system_data)
        
        # Add containment relationships
        for contained in data.get('contains', []):
            if contained in self.formal_systems:
                self.add_relationship(name, contained, 'contains')
    
    def add_relationship(self, source: str, target: str, rel_type: str) -> None:
        """Add a relationship between statements."""
        if source not in self.statements:
            raise ValueError(f"Source statement {source} does not exist")
        if target not in self.statements:
            raise ValueError(f"Target statement {target} does not exist")
            
        relation = EntailmentRelation(
            source=source,
            target=target,
            relation_type=rel_type
        )
        
        self.relations.append(relation)
        self.graph.add_edge(source, target, type=rel_type)
        self.metadata['relation_count'] += 1
        self.metadata['last_modified'] = datetime.now()
    
    def get_statement(self, name: str) -> Optional[LogicalStatement]:
        """Get a statement by name."""
        return self.statements.get(name)
    
    def get_formal_system(self, name: str) -> Optional[Dict]:
        """Get a formal system by name."""
        return self.formal_systems.get(name)
    
    def get_relationships(self, statement: str) -> List[EntailmentRelation]:
        """Get all relationships involving a statement."""
        return [rel for rel in self.relations if rel.source == statement or rel.target == statement]
    
    def to_dict(self) -> Dict:
        """Convert the cone to a dictionary representation."""
        return {
            'name': self.name,
            'statements': {k: v.to_dict() for k, v in self.statements.items()},
            'relations': [rel.to_dict() for rel in self.relations],
            'formal_systems': self.formal_systems,
            'metadata': self.metadata
        }
    
    def validate_structure(self) -> Dict:
        """Validate the cone's structure."""
        validation = {
            'is_valid': True,
            'messages': [],
            'metrics': {
                'statement_count': len(self.statements),
                'relation_count': len(self.relations),
                'formal_systems': list(self.formal_systems.keys()),
                'theorem_count': sum(1 for s in self.statements.values() if s.statement_type == 'theorem'),
                'axiom_count': sum(1 for s in self.statements.values() if s.is_axiom),
                'conjecture_count': sum(1 for s in self.statements.values() if s.statement_type == 'conjecture')
            }
        }
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                validation['is_valid'] = False
                validation['messages'].append(f"Found cycles in the graph: {cycles}")
        except Exception as e:
            validation['is_valid'] = False
            validation['messages'].append(f"Error checking for cycles: {str(e)}")
        
        # Check for disconnected components
        components = list(nx.connected_components(self.graph.to_undirected()))
        if len(components) > 1:
            validation['messages'].append(f"Warning: Graph has {len(components)} disconnected components")
        
        self.metadata['validation_status'] = 'valid' if validation['is_valid'] else 'invalid'
        return validation