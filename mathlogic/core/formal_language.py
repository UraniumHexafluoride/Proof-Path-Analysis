"""
Formal language module for mathematical logic.
Defines the syntax and semantics of our formal language.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Set, Dict, Optional, Union

class TermType(Enum):
    VARIABLE = auto()
    CONSTANT = auto()
    FUNCTION = auto()

class FormulaType(Enum):
    ATOMIC = auto()
    NEGATION = auto()
    CONJUNCTION = auto()
    DISJUNCTION = auto()
    IMPLICATION = auto()
    UNIVERSAL = auto()
    EXISTENTIAL = auto()

@dataclass
class Term:
    """Represents a term in first-order logic."""
    type: TermType
    symbol: str
    arguments: List['Term'] = None

    def __str__(self) -> str:
        if self.type in (TermType.VARIABLE, TermType.CONSTANT):
            return self.symbol
        return f"{self.symbol}({', '.join(str(arg) for arg in self.arguments)})"

@dataclass
class Formula:
    """Represents a formula in first-order logic."""
    type: FormulaType
    symbol: str = ""  # For atomic formulas
    left: Optional[Union['Formula', Term]] = None
    right: Optional[Union['Formula', Term]] = None
    variable: Optional[str] = None  # For quantifiers
    terms: List[Term] = None  # For atomic formulas with terms

    def __str__(self) -> str:
        if self.type == FormulaType.ATOMIC:
            if self.terms:
                terms_str = ', '.join(str(term) for term in self.terms)
                return f"{self.symbol}({terms_str})"
            return self.symbol
        elif self.type == FormulaType.NEGATION:
            return f"¬({str(self.left)})"
        elif self.type == FormulaType.CONJUNCTION:
            return f"({str(self.left)} ∧ {str(self.right)})"
        elif self.type == FormulaType.DISJUNCTION:
            return f"({str(self.left)} ∨ {str(self.right)})"
        elif self.type == FormulaType.IMPLICATION:
            return f"({str(self.left)} → {str(self.right)})"
        elif self.type == FormulaType.UNIVERSAL:
            return f"∀{self.variable}.({str(self.left)})"
        elif self.type == FormulaType.EXISTENTIAL:
            return f"∃{self.variable}.({str(self.left)})"
        return ""

class Parser:
    """Parser for the formal language."""
    
    def __init__(self):
        self.variables = set()
        self.constants = set()
        self.functions = dict()  # name -> arity
        self.predicates = dict()  # name -> arity
        
    def declare_variable(self, name: str) -> None:
        """Declare a variable symbol."""
        self.variables.add(name)
        
    def declare_constant(self, name: str) -> None:
        """Declare a constant symbol."""
        self.constants.add(name)
        
    def declare_function(self, name: str, arity: int) -> None:
        """Declare a function symbol with its arity."""
        self.functions[name] = arity
        
    def declare_predicate(self, name: str, arity: int) -> None:
        """Declare a predicate symbol with its arity."""
        self.predicates[name] = arity
        
    def parse_term(self, input_str: str) -> Term:
        """Parse a string into a Term."""
        input_str = input_str.strip()
        
        # Check if it's a function application
        if '(' in input_str:
            fname = input_str[:input_str.index('(')].strip()
            if fname not in self.functions:
                raise ValueError(f"Undefined function: {fname}")
                
            # Parse arguments
            args_str = input_str[input_str.index('(')+1:input_str.rindex(')')]
            args = [self.parse_term(arg.strip()) for arg in args_str.split(',')]
            
            if len(args) != self.functions[fname]:
                raise ValueError(f"Wrong number of arguments for function {fname}")
                
            return Term(TermType.FUNCTION, fname, args)
            
        # Must be a variable or constant
        if input_str in self.variables:
            return Term(TermType.VARIABLE, input_str)
        elif input_str in self.constants:
            return Term(TermType.CONSTANT, input_str)
        else:
            raise ValueError(f"Undefined symbol: {input_str}")
            
    def parse_formula(self, input_str: str) -> Formula:
        """Parse a string into a Formula."""
        input_str = input_str.strip()
        
        # Handle quantifiers
        if input_str.startswith('∀') or input_str.startswith('∃'):
            quantifier = input_str[0]
            var_end = input_str.index('.')
            var = input_str[1:var_end].strip()
            subformula = input_str[var_end+1:].strip()
            
            if quantifier == '∀':
                return Formula(FormulaType.UNIVERSAL, variable=var, 
                            left=self.parse_formula(subformula))
            else:
                return Formula(FormulaType.EXISTENTIAL, variable=var,
                            left=self.parse_formula(subformula))
                            
        # Handle binary connectives
        if '→' in input_str:
            left, right = input_str.split('→', 1)
            return Formula(FormulaType.IMPLICATION,
                         left=self.parse_formula(left.strip()),
                         right=self.parse_formula(right.strip()))
                         
        if '∨' in input_str:
            left, right = input_str.split('∨', 1)
            return Formula(FormulaType.DISJUNCTION,
                         left=self.parse_formula(left.strip()),
                         right=self.parse_formula(right.strip()))
                         
        if '∧' in input_str:
            left, right = input_str.split('∧', 1)
            return Formula(FormulaType.CONJUNCTION,
                         left=self.parse_formula(left.strip()),
                         right=self.parse_formula(right.strip()))
                         
        # Handle negation
        if input_str.startswith('¬'):
            return Formula(FormulaType.NEGATION,
                         left=self.parse_formula(input_str[1:].strip()))
                         
        # Must be atomic formula
        if '(' in input_str:
            pred = input_str[:input_str.index('(')].strip()
            if pred not in self.predicates:
                raise ValueError(f"Undefined predicate: {pred}")
                
            # Parse terms
            terms_str = input_str[input_str.index('(')+1:input_str.rindex(')')]
            terms = [self.parse_term(term.strip()) for term in terms_str.split(',')]
            
            if len(terms) != self.predicates[pred]:
                raise ValueError(f"Wrong number of arguments for predicate {pred}")
                
            return Formula(FormulaType.ATOMIC, symbol=pred, terms=terms)
            
        # Must be propositional variable
        return Formula(FormulaType.ATOMIC, symbol=input_str) 