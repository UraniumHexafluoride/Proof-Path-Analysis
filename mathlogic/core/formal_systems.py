"""
Formal systems module defining axiom schemas and formal systems.
"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field
from .formal_language import Formula, FormulaType, Term, TermType, Parser

@dataclass
class AxiomSchema:
    """
    Represents an axiom schema that can generate multiple axioms
    through substitution.
    """
    name: str
    pattern: str  # Pattern string in formal language
    description: str
    variables: List[str]  # Free variables that can be substituted
    
    def generate_instance(self, parser: Parser, substitutions: Dict[str, str]) -> Formula:
        """Generate a specific instance of this schema with given substitutions."""
        # Replace variables in pattern with substitutions
        instance_str = self.pattern
        for var, subst in substitutions.items():
            if var in self.variables:
                instance_str = instance_str.replace(var, subst)
        return parser.parse_formula(instance_str)

@dataclass
class FormalSystem:
    """
    Represents a formal system with its axioms and rules.
    """
    name: str
    description: str
    language: Parser
    axiom_schemas: List[AxiomSchema]
    axioms: Set[Formula] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize the formal system."""
        self.axioms = set()
        
    def add_axiom_schema(self, schema: AxiomSchema) -> None:
        """Add an axiom schema to the system."""
        self.axiom_schemas.append(schema)
        
    def add_axiom(self, axiom: Formula) -> None:
        """Add a specific axiom to the system."""
        self.axioms.add(axiom)
        
    def is_axiom(self, formula: Formula) -> bool:
        """Check if a formula is an axiom of the system."""
        # Check direct axioms
        if formula in self.axioms:
            return True
            
        # Check if it's an instance of any schema
        return any(self._is_schema_instance(formula, schema) 
                  for schema in self.axiom_schemas)
                  
    def _is_schema_instance(self, formula: Formula, schema: AxiomSchema) -> bool:
        """Check if a formula is an instance of an axiom schema."""
        # This is a simplified check - a full implementation would need
        # to handle all possible substitutions
        try:
            pattern = self.language.parse_formula(schema.pattern)
            return self._matches_pattern(formula, pattern, {})
        except:
            return False
            
    def _matches_pattern(self, formula: Formula, pattern: Formula,
                        bindings: Dict[str, Formula]) -> bool:
        """
        Check if a formula matches a pattern with given variable bindings.
        This is a simplified unification algorithm.
        """
        if pattern.type == FormulaType.ATOMIC and pattern.symbol in bindings:
            return formula == bindings[pattern.symbol]
            
        if pattern.type != formula.type:
            return False
            
        if pattern.type == FormulaType.ATOMIC:
            if pattern.symbol in self.language.variables:
                bindings[pattern.symbol] = formula
                return True
            return pattern.symbol == formula.symbol
            
        if pattern.type == FormulaType.NEGATION:
            return self._matches_pattern(formula.left, pattern.left, bindings)
            
        if pattern.type in (FormulaType.CONJUNCTION, FormulaType.DISJUNCTION,
                          FormulaType.IMPLICATION):
            return (self._matches_pattern(formula.left, pattern.left, bindings) and
                   self._matches_pattern(formula.right, pattern.right, bindings))
                   
        if pattern.type in (FormulaType.UNIVERSAL, FormulaType.EXISTENTIAL):
            return (pattern.variable == formula.variable and
                   self._matches_pattern(formula.left, pattern.left, bindings))
                   
        return False

# Define some common axiom schemas for propositional logic
PROPOSITIONAL_AXIOMS = [
    AxiomSchema(
        "Modus Ponens Schema",
        "(A → B) → (A → B)",
        "If A implies B, then A implies B",
        ["A", "B"]
    ),
    AxiomSchema(
        "And Introduction Schema",
        "A → (B → (A ∧ B))",
        "If A and B are true, then their conjunction is true",
        ["A", "B"]
    ),
    AxiomSchema(
        "And Elimination Schema 1",
        "(A ∧ B) → A",
        "If a conjunction is true, its first part is true",
        ["A", "B"]
    ),
    AxiomSchema(
        "And Elimination Schema 2",
        "(A ∧ B) → B",
        "If a conjunction is true, its second part is true",
        ["A", "B"]
    ),
    AxiomSchema(
        "Or Introduction Schema 1",
        "A → (A ∨ B)",
        "If A is true, then A or B is true",
        ["A", "B"]
    ),
    AxiomSchema(
        "Or Introduction Schema 2",
        "B → (A ∨ B)",
        "If B is true, then A or B is true",
        ["A", "B"]
    )
]

# Define some common axiom schemas for first-order logic
FIRST_ORDER_AXIOMS = PROPOSITIONAL_AXIOMS + [
    AxiomSchema(
        "Universal Instantiation",
        "∀x.A(x) → A(t)",
        "If a property holds for all x, it holds for any term t",
        ["A", "x", "t"]
    ),
    AxiomSchema(
        "Universal Generalization",
        "A(x) → ∀x.A(x)",
        "If a property holds for any arbitrary x, it holds for all x",
        ["A", "x"]
    ),
    AxiomSchema(
        "Existential Introduction",
        "A(t) → ∃x.A(x)",
        "If a property holds for some term t, then there exists an x with that property",
        ["A", "x", "t"]
    ),
    AxiomSchema(
        "Existential Elimination",
        "(∃x.A(x) ∧ (∀y.(A(y) → B))) → B",
        "If something exists with property A, and A always implies B, then B is true",
        ["A", "B", "x", "y"]
    )
]

def create_propositional_logic() -> FormalSystem:
    """Create a formal system for propositional logic."""
    parser = Parser()
    # Add basic propositional variables
    for var in ["A", "B", "C", "P", "Q", "R"]:
        parser.declare_variable(var)
    
    return FormalSystem(
        "Propositional Logic",
        "Classical propositional logic with standard axioms",
        parser,
        PROPOSITIONAL_AXIOMS
    )

def create_first_order_logic() -> FormalSystem:
    """Create a formal system for first-order logic."""
    parser = Parser()
    # Add variables and predicates
    for var in ["x", "y", "z"]:
        parser.declare_variable(var)
    for pred in ["P", "Q", "R"]:
        parser.declare_predicate(pred, 1)  # Unary predicates
    
    return FormalSystem(
        "First-Order Logic",
        "Classical first-order logic with standard axioms",
        parser,
        FIRST_ORDER_AXIOMS
    ) 