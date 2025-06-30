"""
Tests for the theorem prover implementation.
"""

import unittest
from ..core.formal_language import Formula, FormulaType, Parser
from ..core.proof_checker import ProofChecker
from ..core.formal_systems import create_propositional_logic, create_first_order_logic
from ..core.theorem_prover import TheoremProver, Resolution

class TestTheoremProver(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.prop_system = create_propositional_logic()
        self.fol_system = create_first_order_logic()
        self.checker = ProofChecker()
        self.prover = TheoremProver(self.prop_system, self.checker)
        self.resolution = Resolution()
        
    def test_modus_ponens(self):
        """Test Modus Ponens inference."""
        # Create formulas A and (A → B)
        parser = self.prop_system.language
        a = Formula(FormulaType.ATOMIC, symbol="A")
        b = Formula(FormulaType.ATOMIC, symbol="B")
        impl = Formula(FormulaType.IMPLICATION, left=a, right=b)
        
        # Add as axioms
        self.prop_system.add_axiom(a)
        self.prop_system.add_axiom(impl)
        
        # Try to prove B
        proof = self.prover.prove(b)
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula, b)
        
    def test_and_introduction(self):
        """Test And-Introduction inference."""
        parser = self.prop_system.language
        a = Formula(FormulaType.ATOMIC, symbol="A")
        b = Formula(FormulaType.ATOMIC, symbol="B")
        conj = Formula(FormulaType.CONJUNCTION, left=a, right=b)
        
        # Add premises as axioms
        self.prop_system.add_axiom(a)
        self.prop_system.add_axiom(b)
        
        # Try to prove (A ∧ B)
        proof = self.prover.prove(conj)
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula, conj)
        
    def test_and_elimination(self):
        """Test And-Elimination inference."""
        parser = self.prop_system.language
        a = Formula(FormulaType.ATOMIC, symbol="A")
        b = Formula(FormulaType.ATOMIC, symbol="B")
        conj = Formula(FormulaType.CONJUNCTION, left=a, right=b)
        
        # Add conjunction as axiom
        self.prop_system.add_axiom(conj)
        
        # Try to prove A and B separately
        proof_a = self.prover.prove(a)
        proof_b = self.prover.prove(b)
        
        self.assertIsNotNone(proof_a)
        self.assertIsNotNone(proof_b)
        self.assertEqual(proof_a[-1].formula, a)
        self.assertEqual(proof_b[-1].formula, b)
        
    def test_or_introduction(self):
        """Test Or-Introduction inference."""
        parser = self.prop_system.language
        a = Formula(FormulaType.ATOMIC, symbol="A")
        b = Formula(FormulaType.ATOMIC, symbol="B")
        disj = Formula(FormulaType.DISJUNCTION, left=a, right=b)
        
        # Add A as axiom
        self.prop_system.add_axiom(a)
        
        # Try to prove (A ∨ B)
        proof = self.prover.prove(disj)
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula, disj)
        
    def test_resolution(self):
        """Test Resolution-based proving."""
        parser = self.prop_system.language
        a = Formula(FormulaType.ATOMIC, symbol="A")
        not_a = Formula(FormulaType.NEGATION, left=a)
        
        # Try to prove contradiction from A and ¬A
        axioms = {a, not_a}
        proof = self.resolution.prove_by_resolution(axioms, 
            Formula(FormulaType.ATOMIC, symbol="⊥"))
            
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula.symbol, "⊥")
        
    def test_first_order_logic(self):
        """Test first-order logic proving."""
        parser = self.fol_system.language
        
        # Test ∀x.P(x) → P(t)
        parser.declare_variable("x")
        parser.declare_predicate("P", 1)
        parser.declare_constant("t")
        
        forall = Formula(FormulaType.UNIVERSAL, variable="x",
                        left=Formula(FormulaType.ATOMIC, symbol="P",
                                   terms=[Term(TermType.VARIABLE, "x")]))
                                   
        instance = Formula(FormulaType.ATOMIC, symbol="P",
                         terms=[Term(TermType.CONSTANT, "t")])
                         
        impl = Formula(FormulaType.IMPLICATION, left=forall, right=instance)
        
        # Add universal formula as axiom
        self.fol_system.add_axiom(forall)
        
        # Try to prove P(t)
        proof = self.prover.prove(instance)
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula, instance)
        
    def test_complex_proof(self):
        """Test a more complex proof involving multiple steps."""
        parser = self.prop_system.language
        
        # Set up formulas for ((A → B) ∧ (B → C)) → (A → C)
        a = Formula(FormulaType.ATOMIC, symbol="A")
        b = Formula(FormulaType.ATOMIC, symbol="B")
        c = Formula(FormulaType.ATOMIC, symbol="C")
        
        ab = Formula(FormulaType.IMPLICATION, left=a, right=b)
        bc = Formula(FormulaType.IMPLICATION, left=b, right=c)
        ac = Formula(FormulaType.IMPLICATION, left=a, right=c)
        
        premises = Formula(FormulaType.CONJUNCTION, left=ab, right=bc)
        
        # Add premises as axiom
        self.prop_system.add_axiom(premises)
        self.prop_system.add_axiom(a)
        
        # Try to prove C
        proof = self.prover.prove(c)
        self.assertIsNotNone(proof)
        self.assertEqual(proof[-1].formula, c)

if __name__ == '__main__':
    unittest.main() 