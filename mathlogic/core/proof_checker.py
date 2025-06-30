"""
Proof checker module for verifying formal proofs.
Implements inference rules and proof verification.
"""

from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from .formal_language import Formula, FormulaType, Term, TermType

@dataclass
class ProofStep:
    """Represents a single step in a formal proof."""
    formula: Formula
    rule: str
    premises: List[int]  # Indices of premises used in this step
    explanation: str

class InferenceRule:
    """Base class for inference rules."""
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, premises: List[Formula]) -> Optional[Formula]:
        """Apply the rule to the given premises."""
        raise NotImplementedError

class ModusPonens(InferenceRule):
    """Modus Ponens rule: from A and A → B, derive B."""
    def __init__(self):
        super().__init__("Modus Ponens")
    
    def apply(self, premises: List[Formula]) -> Optional[Formula]:
        if len(premises) != 2:
            return None
            
        # Find implication and antecedent
        impl = next((p for p in premises if p.type == FormulaType.IMPLICATION), None)
        if not impl:
            return None
            
        # Check if other premise matches antecedent
        antecedent = next((p for p in premises if p != impl), None)
        if not antecedent or antecedent != impl.left:
            return None
            
        return impl.right

class AndIntroduction(InferenceRule):
    """And-Introduction: from A and B, derive A ∧ B."""
    def __init__(self):
        super().__init__("And-Introduction")
    
    def apply(self, premises: List[Formula]) -> Optional[Formula]:
        if len(premises) != 2:
            return None
        return Formula(FormulaType.CONJUNCTION, left=premises[0], right=premises[1])

class AndElimination(InferenceRule):
    """And-Elimination: from A ∧ B, derive A or B."""
    def __init__(self):
        super().__init__("And-Elimination")
    
    def apply(self, premises: List[Formula]) -> Optional[Formula]:
        if len(premises) != 1 or premises[0].type != FormulaType.CONJUNCTION:
            return None
        # Return both conjuncts (caller will choose which one to use)
        return [premises[0].left, premises[0].right]

class OrIntroduction(InferenceRule):
    """Or-Introduction: from A, derive A ∨ B (or B ∨ A)."""
    def __init__(self):
        super().__init__("Or-Introduction")
    
    def apply(self, premises: List[Formula]) -> Optional[Formula]:
        if len(premises) != 1:
            return None
        # Note: This creates a disjunction with a "hole" for the other disjunct
        return Formula(FormulaType.DISJUNCTION, left=premises[0], right=None)

class ImplicationIntroduction(InferenceRule):
    """
    Implication-Introduction: If assuming A leads to B, derive A → B.
    This is a bit special as it requires tracking assumptions.
    """
    def __init__(self):
        super().__init__("Implication-Introduction")
    
    def apply(self, premises: List[Formula], assumption: Formula = None) -> Optional[Formula]:
        if len(premises) != 1 or not assumption:
            return None
        return Formula(FormulaType.IMPLICATION, left=assumption, right=premises[0])

class ProofChecker:
    """Verifies formal proofs step by step."""
    
    def __init__(self):
        self.rules = [
            ModusPonens(),
            AndIntroduction(),
            AndElimination(),
            OrIntroduction(),
            ImplicationIntroduction()
        ]
        
    def verify_step(self, step: ProofStep, previous_steps: List[ProofStep]) -> bool:
        """
        Verify that a proof step follows from previous steps using valid rules.
        
        Args:
            step: The proof step to verify
            previous_steps: List of all previous steps in the proof
            
        Returns:
            bool: Whether the step is valid
        """
        # Check that all premise indices are valid
        if any(i >= len(previous_steps) for i in step.premises):
            return False
            
        # Get the actual premises
        premises = [previous_steps[i].formula for i in step.premises]
        
        # Try each rule
        for rule in self.rules:
            if rule.name == step.rule:
                result = rule.apply(premises)
                if isinstance(result, list):
                    # Handle rules that can produce multiple results (like And-Elimination)
                    if step.formula in result:
                        return True
                elif result == step.formula:
                    return True
                break
                
        return False
        
    def verify_proof(self, steps: List[ProofStep], assumptions: Set[Formula] = None) -> bool:
        """
        Verify an entire proof.
        
        Args:
            steps: List of proof steps
            assumptions: Set of formulas we're allowed to assume (optional)
            
        Returns:
            bool: Whether the proof is valid
        """
        assumptions = assumptions or set()
        
        for i, step in enumerate(steps):
            # Skip verification for assumptions
            if step.formula in assumptions and not step.premises:
                continue
                
            if not self.verify_step(step, steps[:i]):
                return False
                
        return True

class ProofGenerator:
    """
    Attempts to generate proofs automatically.
    This is a simple implementation - for real theorem proving,
    you'd want to use more sophisticated algorithms.
    """
    
    def __init__(self):
        self.checker = ProofChecker()
        
    def find_proof(self, goal: Formula, assumptions: Set[Formula],
                  max_depth: int = 5) -> Optional[List[ProofStep]]:
        """
        Try to find a proof of the goal from the assumptions.
        Uses a simple forward-chaining approach.
        
        Args:
            goal: The formula to prove
            assumptions: Set of formulas we can use
            max_depth: Maximum number of steps to try
            
        Returns:
            List[ProofStep] if a proof is found, None otherwise
        """
        if goal in assumptions:
            return [ProofStep(goal, "Assumption", [], "Given assumption")]
            
        current_formulas = set(assumptions)
        proof_steps = []
        
        # Try applying rules forward
        for _ in range(max_depth):
            new_formulas = set()
            
            # Try each rule with each combination of current formulas
            for rule in self.checker.rules:
                for premises in self._get_premise_combinations(current_formulas, rule):
                    result = rule.apply(premises)
                    
                    if isinstance(result, list):
                        results = result
                    else:
                        results = [result] if result else []
                        
                    for new_formula in results:
                        if new_formula and new_formula not in current_formulas:
                            new_formulas.add(new_formula)
                            premise_indices = [
                                i for i, step in enumerate(proof_steps)
                                if step.formula in premises
                            ]
                            proof_steps.append(ProofStep(
                                new_formula,
                                rule.name,
                                premise_indices,
                                f"Applied {rule.name}"
                            ))
                            
                            if new_formula == goal:
                                return proof_steps
            
            if not new_formulas:
                break
                
            current_formulas.update(new_formulas)
            
        return None
        
    def _get_premise_combinations(self, formulas: Set[Formula], rule: InferenceRule) -> List[List[Formula]]:
        """Get all possible combinations of premises for a rule."""
        if rule.name == "Modus Ponens":
            result = []
            for f1 in formulas:
                for f2 in formulas:
                    if f2.type == FormulaType.IMPLICATION and f2.left == f1:
                        result.append([f1, f2])
            return result
        elif rule.name == "And-Introduction":
            return [[f1, f2] for f1 in formulas for f2 in formulas if f1 != f2]
        elif rule.name == "And-Elimination":
            return [[f] for f in formulas if f.type == FormulaType.CONJUNCTION]
        elif rule.name == "Or-Introduction":
            return [[f] for f in formulas]
        elif rule.name == "Implication-Introduction":
            # This is trickier and requires assumption tracking
            return []
        return [] 