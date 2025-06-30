"""
Theorem prover module implementing automated theorem proving capabilities.
"""

from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from .formal_language import Formula, FormulaType, Term, TermType
from .proof_checker import ProofStep, ProofChecker
from .formal_systems import FormalSystem

@dataclass
class ProofState:
    """Represents the state of a proof attempt."""
    goal: Formula
    assumptions: Set[Formula]
    proven: Set[Formula]
    steps: List[ProofStep]
    depth: int = 0

class TheoremProver:
    """
    Automated theorem prover using a combination of forward and backward chaining.
    """
    
    def __init__(self, system: FormalSystem, checker: ProofChecker):
        self.system = system
        self.checker = checker
        
    def prove(self, goal: Formula, max_depth: int = 5) -> Optional[List[ProofStep]]:
        """
        Try to prove a goal formula using the axioms and rules of the system.
        
        Args:
            goal: The formula to prove
            max_depth: Maximum search depth
            
        Returns:
            List[ProofStep] if a proof is found, None otherwise
        """
        # Start with axioms as assumptions
        assumptions = self.system.axioms.copy()
        
        # Create initial state
        state = ProofState(
            goal=goal,
            assumptions=assumptions,
            proven=set(assumptions),
            steps=[]
        )
        
        # Try to find a proof using bidirectional search
        return self._prove_bidirectional(state, max_depth)
        
    def _prove_bidirectional(self, state: ProofState, max_depth: int) -> Optional[List[ProofStep]]:
        """
        Try to prove the goal using bidirectional search.
        Combines forward chaining from assumptions and backward chaining from goal.
        """
        if state.depth > max_depth:
            return None
            
        # Check if goal is already proven
        if state.goal in state.proven:
            return state.steps
            
        # Try forward chaining
        forward_steps = self._forward_chain(state)
        if forward_steps:
            return forward_steps
            
        # Try backward chaining
        backward_steps = self._backward_chain(state)
        if backward_steps:
            return backward_steps
            
        return None
        
    def _forward_chain(self, state: ProofState) -> Optional[List[ProofStep]]:
        """Apply rules forward from assumptions."""
        new_formulas = set()
        
        # Try each rule with current proven formulas
        for rule in self.checker.rules:
            for premises in self._get_premise_combinations(state.proven, rule):
                result = rule.apply(premises)
                
                if isinstance(result, list):
                    results = result
                else:
                    results = [result] if result else []
                    
                for new_formula in results:
                    if new_formula and new_formula not in state.proven:
                        new_formulas.add(new_formula)
                        premise_indices = [
                            i for i, step in enumerate(state.steps)
                            if step.formula in premises
                        ]
                        state.steps.append(ProofStep(
                            new_formula,
                            rule.name,
                            premise_indices,
                            f"Applied {rule.name}"
                        ))
                        
                        if new_formula == state.goal:
                            return state.steps
        
        if new_formulas:
            # Update state and continue search
            state.proven.update(new_formulas)
            state.depth += 1
            return self._prove_bidirectional(state, state.depth + 5)
            
        return None
        
    def _backward_chain(self, state: ProofState) -> Optional[List[ProofStep]]:
        """Work backward from goal to find a proof."""
        # Look for rules that could produce the goal
        for rule in self.checker.rules:
            possible_premises = self._find_possible_premises(state.goal, rule)
            
            for premises in possible_premises:
                # Check if we can prove all premises
                all_proven = True
                subproofs = []
                
                for premise in premises:
                    if premise not in state.proven:
                        # Recursively try to prove the premise
                        substate = ProofState(
                            goal=premise,
                            assumptions=state.assumptions,
                            proven=state.proven,
                            steps=state.steps.copy(),
                            depth=state.depth + 1
                        )
                        
                        subproof = self._prove_bidirectional(substate, state.depth + 5)
                        if not subproof:
                            all_proven = False
                            break
                        subproofs.extend(subproof)
                
                if all_proven:
                    # We found a way to prove all premises
                    state.steps.extend(subproofs)
                    
                    # Add the final step using the rule
                    premise_indices = [
                        i for i, step in enumerate(state.steps)
                        if step.formula in premises
                    ]
                    state.steps.append(ProofStep(
                        state.goal,
                        rule.name,
                        premise_indices,
                        f"Applied {rule.name}"
                    ))
                    
                    return state.steps
        
        return None
        
    def _get_premise_combinations(self, formulas: Set[Formula], rule: str) -> List[List[Formula]]:
        """Get all possible combinations of premises for a rule."""
        # Reuse the implementation from ProofGenerator
        return self.checker._get_premise_combinations(formulas, rule)
        
    def _find_possible_premises(self, goal: Formula, rule: str) -> List[List[Formula]]:
        """Find possible premises that could lead to the goal using the rule."""
        if rule.name == "Modus Ponens":
            # Goal is B, need A and (A → B)
            # Return a template - actual A needs to be found
            return [[Formula(FormulaType.ATOMIC, symbol="?"),
                    Formula(FormulaType.IMPLICATION, left=Formula(FormulaType.ATOMIC, symbol="?"),
                           right=goal)]]
                           
        elif rule.name == "And-Introduction":
            # Goal is (A ∧ B), need A and B
            if goal.type == FormulaType.CONJUNCTION:
                return [[goal.left, goal.right]]
                
        elif rule.name == "And-Elimination":
            # Can't work backward from this
            return []
            
        elif rule.name == "Or-Introduction":
            # Goal is (A ∨ B), need either A or B
            if goal.type == FormulaType.DISJUNCTION:
                return [[goal.left], [goal.right]]
                
        elif rule.name == "Implication-Introduction":
            # Goal is (A → B), need B under assumption A
            if goal.type == FormulaType.IMPLICATION:
                return [[goal.right]]  # Note: need to add goal.left to assumptions
                
        return []

class Resolution:
    """
    Implementation of the resolution method for theorem proving.
    This is particularly effective for propositional logic.
    """
    
    @staticmethod
    def negate(formula: Formula) -> Formula:
        """Create the negation of a formula."""
        if formula.type == FormulaType.NEGATION:
            return formula.left  # Double negation elimination
        return Formula(FormulaType.NEGATION, left=formula)
        
    @staticmethod
    def to_cnf(formula: Formula) -> Formula:
        """Convert a formula to Conjunctive Normal Form (CNF)."""
        # This is a simplified implementation
        # A full implementation would handle all cases
        
        # Remove implications
        formula = Resolution._remove_implications(formula)
        
        # Push negations inward
        formula = Resolution._push_negations(formula)
        
        # Distribute OR over AND
        formula = Resolution._distribute_or(formula)
        
        return formula
        
    @staticmethod
    def _remove_implications(formula: Formula) -> Formula:
        """Replace A → B with ¬A ∨ B."""
        if formula.type == FormulaType.IMPLICATION:
            return Formula(
                FormulaType.DISJUNCTION,
                left=Formula(FormulaType.NEGATION, left=formula.left),
                right=formula.right
            )
            
        if formula.type == FormulaType.NEGATION:
            return Formula(FormulaType.NEGATION,
                         left=Resolution._remove_implications(formula.left))
                         
        if formula.type in (FormulaType.CONJUNCTION, FormulaType.DISJUNCTION):
            return Formula(
                formula.type,
                left=Resolution._remove_implications(formula.left),
                right=Resolution._remove_implications(formula.right)
            )
            
        return formula
        
    @staticmethod
    def _push_negations(formula: Formula) -> Formula:
        """Push negations inward using De Morgan's laws."""
        if formula.type != FormulaType.NEGATION:
            return formula
            
        if formula.left.type == FormulaType.CONJUNCTION:
            # ¬(A ∧ B) = ¬A ∨ ¬B
            return Formula(
                FormulaType.DISJUNCTION,
                left=Resolution._push_negations(Formula(FormulaType.NEGATION,
                                                      left=formula.left.left)),
                right=Resolution._push_negations(Formula(FormulaType.NEGATION,
                                                       left=formula.left.right))
            )
            
        if formula.left.type == FormulaType.DISJUNCTION:
            # ¬(A ∨ B) = ¬A ∧ ¬B
            return Formula(
                FormulaType.CONJUNCTION,
                left=Resolution._push_negations(Formula(FormulaType.NEGATION,
                                                      left=formula.left.left)),
                right=Resolution._push_negations(Formula(FormulaType.NEGATION,
                                                       left=formula.left.right))
            )
            
        return formula
        
    @staticmethod
    def _distribute_or(formula: Formula) -> Formula:
        """Distribute OR over AND: A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)."""
        if formula.type != FormulaType.DISJUNCTION:
            return formula
            
        if formula.right.type == FormulaType.CONJUNCTION:
            return Formula(
                FormulaType.CONJUNCTION,
                left=Formula(FormulaType.DISJUNCTION,
                           left=formula.left,
                           right=formula.right.left),
                right=Formula(FormulaType.DISJUNCTION,
                            left=formula.left,
                            right=formula.right.right)
            )
            
        if formula.left.type == FormulaType.CONJUNCTION:
            return Formula(
                FormulaType.CONJUNCTION,
                left=Formula(FormulaType.DISJUNCTION,
                           left=formula.left.left,
                           right=formula.right),
                right=Formula(FormulaType.DISJUNCTION,
                            left=formula.left.right,
                            right=formula.right)
            )
            
        return formula
        
    def prove_by_resolution(self, axioms: Set[Formula], goal: Formula) -> Optional[List[ProofStep]]:
        """
        Try to prove a goal using resolution.
        Returns a proof if successful, None otherwise.
        """
        # Convert everything to CNF
        cnf_axioms = {self.to_cnf(axiom) for axiom in axioms}
        cnf_goal = self.to_cnf(goal)
        
        # Add negation of goal (for proof by contradiction)
        clauses = cnf_axioms | {self.negate(cnf_goal)}
        
        # Try to derive empty clause (contradiction)
        steps = []
        seen = set()
        
        while True:
            new_clauses = set()
            
            # Try all pairs of clauses
            for c1 in clauses:
                for c2 in clauses:
                    if c1 != c2:
                        resolvents = self._resolve(c1, c2)
                        
                        for resolvent in resolvents:
                            if not resolvent.symbol:  # Empty clause
                                # We found a contradiction!
                                steps.append(ProofStep(
                                    Formula(FormulaType.ATOMIC, symbol="⊥"),
                                    "Resolution",
                                    [],
                                    "Derived contradiction"
                                ))
                                return steps
                                
                            if resolvent not in seen:
                                new_clauses.add(resolvent)
                                seen.add(resolvent)
                                steps.append(ProofStep(
                                    resolvent,
                                    "Resolution",
                                    [],
                                    f"Resolved {c1} and {c2}"
                                ))
            
            if not new_clauses - clauses:
                # No new clauses derived
                return None
                
            clauses |= new_clauses
            
    def _resolve(self, c1: Formula, c2: Formula) -> Set[Formula]:
        """
        Apply resolution rule to two clauses.
        Returns set of possible resolvents.
        """
        # This is a simplified implementation
        # A full implementation would handle all cases
        resolvents = set()
        
        if c1.type == FormulaType.ATOMIC and c2.type == FormulaType.NEGATION:
            if c1 == c2.left:
                resolvents.add(Formula(FormulaType.ATOMIC, symbol=""))
                
        elif c2.type == FormulaType.ATOMIC and c1.type == FormulaType.NEGATION:
            if c2 == c1.left:
                resolvents.add(Formula(FormulaType.ATOMIC, symbol=""))
                
        return resolvents 