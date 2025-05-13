"""
Module to create an expanded entailment graph for analysis.
"""

import networkx as nx
import random

def create_expanded_entailment_graph():
    """
    Create an expanded entailment graph with formal systems, theorems,
    and independence relations.
    """
    G = nx.DiGraph()
    
    # Add formal systems
    systems = [
        "ZFC", "ZF", "PA", "PA2", "ACA0", "ZFC+LC", "ZFC+PD", "ZFC+AD", "ZFC+MM"
    ]
    
    for system in systems:
        G.add_node(system, type="system", description=f"Formal system: {system}")
    
    # Add theorems
    theorems = [
        "Continuum Hypothesis", "Axiom of Choice", "Gödel's Incompleteness", 
        "Fermat's Last Theorem", "Four Color Theorem", "Riemann Hypothesis",
        "P vs NP", "Twin Prime Conjecture", "Goldbach's Conjecture",
        "Poincaré Conjecture", "ABC Conjecture", "Collatz Conjecture",
        "Birch and Swinnerton-Dyer Conjecture", "Hodge Conjecture",
        "Navier-Stokes Existence and Smoothness"
    ]
    
    for theorem in theorems:
        G.add_node(theorem, type="theorem", description=f"Mathematical theorem: {theorem}")
    
    # Add entailment relations (system proves theorem)
    entailments = [
        ("ZFC", "Axiom of Choice"),
        ("ZFC+LC", "Continuum Hypothesis"),
        ("PA", "Fermat's Last Theorem"),
        ("ZFC", "Four Color Theorem"),
        ("ZFC+PD", "Poincaré Conjecture"),
        ("PA2", "Gödel's Incompleteness"),
        ("ZFC+AD", "Birch and Swinnerton-Dyer Conjecture"),
        ("ZFC+MM", "Hodge Conjecture")
    ]
    
    for source, target in entailments:
        G.add_edge(source, target, relation="Proves")
    
    # Add independence relations
    independence_relations = [
        ("ZF", "Axiom of Choice"),
        ("ZFC", "Continuum Hypothesis"),
        ("PA", "Gödel's Incompleteness"),
        ("ACA0", "Riemann Hypothesis"),
        ("ZFC", "P vs NP"),
        ("PA", "Twin Prime Conjecture")
    ]
    
    for source, target in independence_relations:
        G.add_edge(source, target, relation="Independence")
    
    # Add some theorem-to-theorem implications
    theorem_implications = [
        ("Riemann Hypothesis", "Twin Prime Conjecture"),
        ("Poincaré Conjecture", "Hodge Conjecture"),
        ("P vs NP", "Collatz Conjecture"),
        ("Goldbach's Conjecture", "Twin Prime Conjecture")
    ]
    
    for source, target in theorem_implications:
        G.add_edge(source, target, relation="Implies")
    
    return G

def save_expanded_graph_to_csv(G, filename="expanded_entailment_graph.csv"):
    """Save the expanded graph to a CSV file."""
    import csv
    import os
    
    # Ensure output directory exists
    os.makedirs("entailment_output", exist_ok=True)
    filepath = os.path.join("entailment_output", filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Source', 'Target', 'Relation'])
        
        for source, target, data in G.edges(data=True):
            relation = data.get('relation', 'Implies')
            writer.writerow([source, target, relation])
    
    print(f"Expanded graph saved to {filepath}")
    return filepath

if __name__ == "__main__":
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Systems: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'system'])}")
    print(f"Theorems: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'theorem'])}")
    
    # Save to CSV
    csv_path = save_expanded_graph_to_csv(G)
    
    print("Expanded entailment graph created successfully!")
