"""
Example script demonstrating how to use the theorem prover.
"""

from mathlogic.core.formal_language import Formula, FormulaType, Term, TermType
from mathlogic.core.proof_checker import ProofChecker
from mathlogic.core.formal_systems import create_propositional_logic, create_first_order_logic
from mathlogic.core.theorem_prover import TheoremProver, Resolution

def prove_transitivity():
    """
    Prove transitivity of implication:
    From (A → B) and (B → C), prove (A → C)
    """
    print("Proving transitivity of implication...")
    
    # Set up the formal system
    system = create_propositional_logic()
    checker = ProofChecker()
    prover = TheoremProver(system, checker)
    
    # Create formulas
    a = Formula(FormulaType.ATOMIC, symbol="A")
    b = Formula(FormulaType.ATOMIC, symbol="B")
    c = Formula(FormulaType.ATOMIC, symbol="C")
    
    ab = Formula(FormulaType.IMPLICATION, left=a, right=b)
    bc = Formula(FormulaType.IMPLICATION, left=b, right=c)
    ac = Formula(FormulaType.IMPLICATION, left=a, right=c)
    
    # Add premises as axioms
    system.add_axiom(ab)
    system.add_axiom(bc)
    
    # Try to prove A → C
    print("\nPremises:")
    print(f"1. {ab}")
    print(f"2. {bc}")
    print(f"\nTrying to prove: {ac}")
    
    proof = prover.prove(ac)
    
    if proof:
        print("\nProof found!")
        print("\nSteps:")
        for i, step in enumerate(proof):
            print(f"{i+1}. {step.formula} ({step.rule})")
    else:
        print("\nNo proof found.")

def prove_contradiction():
    """
    Prove that A and ¬A lead to a contradiction.
    """
    print("\nProving that A and ¬A lead to a contradiction...")
    
    # Set up the formal system
    system = create_propositional_logic()
    checker = ProofChecker()
    resolution = Resolution()
    
    # Create formulas
    a = Formula(FormulaType.ATOMIC, symbol="A")
    not_a = Formula(FormulaType.NEGATION, left=a)
    
    # Try to prove contradiction
    print("\nPremises:")
    print(f"1. {a}")
    print(f"2. {not_a}")
    print("\nTrying to prove contradiction...")
    
    proof = resolution.prove_by_resolution(
        {a, not_a},
        Formula(FormulaType.ATOMIC, symbol="⊥")
    )
    
    if proof:
        print("\nContradiction found!")
        print("\nSteps:")
        for i, step in enumerate(proof):
            print(f"{i+1}. {step.formula} ({step.rule})")
    else:
        print("\nNo contradiction found.")

def prove_universal_instantiation():
    """
    Prove that from ∀x.P(x) we can derive P(t) for any term t.
    """
    print("\nProving universal instantiation...")
    
    # Set up the formal system
    system = create_first_order_logic()
    checker = ProofChecker()
    prover = TheoremProver(system, checker)
    
    # Declare symbols
    parser = system.language
    parser.declare_variable("x")
    parser.declare_predicate("P", 1)
    parser.declare_constant("t")
    
    # Create formulas
    forall = Formula(
        FormulaType.UNIVERSAL,
        variable="x",
        left=Formula(
            FormulaType.ATOMIC,
            symbol="P",
            terms=[Term(TermType.VARIABLE, "x")]
        )
    )
    
    instance = Formula(
        FormulaType.ATOMIC,
        symbol="P",
        terms=[Term(TermType.CONSTANT, "t")]
    )
    
    # Add universal formula as axiom
    system.add_axiom(forall)
    
    # Try to prove P(t)
    print("\nPremise:")
    print(f"1. {forall}")
    print(f"\nTrying to prove: {instance}")
    
    proof = prover.prove(instance)
    
    if proof:
        print("\nProof found!")
        print("\nSteps:")
        for i, step in enumerate(proof):
            print(f"{i+1}. {step.formula} ({step.rule})")
    else:
        print("\nNo proof found.")

if __name__ == "__main__":
    print("Theorem Prover Examples\n")
    print("=" * 50)
    
    prove_transitivity()
    print("\n" + "=" * 50)
    
    prove_contradiction()
    print("\n" + "=" * 50)
    
    prove_universal_instantiation()
    print("\n" + "=" * 50) 