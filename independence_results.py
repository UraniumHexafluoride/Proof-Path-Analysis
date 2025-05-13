import networkx as nx
import csv
import os

# Use absolute path for output directory
OUTPUT_DIR = os.path.abspath("entailment_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_independence_entailment_graph():
    """
    Create an entailment graph with well-known independence results.
    Includes systems like ZFC, PA, and theorems like Paris-Harrington and Goodstein.
    """
    G = nx.DiGraph()
    
    # Add formal systems as nodes
    systems = {
        "ZFC": "Zermelo-Fraenkel set theory with Choice",
        "PA": "Peano Arithmetic",
        "ZF": "Zermelo-Fraenkel set theory without Choice",
        "ZFC+LC": "ZFC with large cardinal axioms",
        "PA+Con(PA)": "PA with the consistency of PA",
        "ACA0": "Arithmetical Comprehension Axiom",
        "RCA0": "Recursive Comprehension Axiom"
    }
    
    for system, description in systems.items():
        G.add_node(system, type="system", description=description)
    
    # Add theorems and statements as nodes
    theorems = {
        "PH": "Paris-Harrington theorem",
        "GT": "Goodstein theorem",
        "CH": "Continuum Hypothesis",
        "AC": "Axiom of Choice",
        "Con(PA)": "Consistency of PA",
        "Con(ZFC)": "Consistency of ZFC",
        "GCH": "Generalized Continuum Hypothesis"
    }
    
    for theorem, description in theorems.items():
        G.add_node(theorem, type="theorem", description=description)
    
    # Add entailment relationships
    entailments = [
        # System containment
        ("ZFC", "ZF", "Contains"),
        ("ZFC", "AC", "Contains"),
        ("ZFC+LC", "ZFC", "Contains"),
        ("PA+Con(PA)", "PA", "Contains"),
        ("PA+Con(PA)", "Con(PA)", "Contains"),
        ("ACA0", "RCA0", "Contains"),
        
        # Provability relationships
        ("ZFC", "Con(PA)", "Proves"),
        ("PA", "PH", "Independence"),  # Paris-Harrington is independent of PA
        ("PA", "GT", "Independence"),  # Goodstein is independent of PA
        ("ZFC", "CH", "Independence"),  # CH is independent of ZFC
        ("ZF", "AC", "Independence"),  # AC is independent of ZF
        ("PA", "Con(PA)", "Independence"),  # Gödel's Second Incompleteness
        ("ZFC", "Con(ZFC)", "Independence"),  # Gödel's Second Incompleteness
        ("ZFC+LC", "CH", "Disproves"),  # Some large cardinals disprove CH
        
        # Reverse mathematics relationships
        ("ACA0", "PH", "Proves"),  # ACA0 proves Paris-Harrington
        ("ACA0", "GT", "Proves"),  # ACA0 proves Goodstein
        
        # Other relationships
        ("ZFC", "GCH", "Independence"),
        ("PA+Con(PA)", "PH", "Proves"),
        ("PA+Con(PA)", "GT", "Proves")
    ]
    
    for source, target, relation in entailments:
        G.add_edge(source, target, relation=relation)
    
    # Save the graph to CSV
    filepath = os.path.join(OUTPUT_DIR, "independence_graph.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Relation"])
        for source, target, data in G.edges(data=True):
            writer.writerow([source, target, data["relation"]])
    
    print(f"Independence entailment graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Saved to {filepath}")
    
    return G

def create_expanded_independence_entailment_graph():
    """Create an expanded entailment graph with more independence results and formal systems."""
    print("Creating expanded independence entailment graph...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add node types
    systems = [
        "ZFC", "ZF", "PA", "PA2", "ACA0", "RCA0", "ZFC+LC", "PA+Con(PA)", 
        "ZF+DC", "ZFC+PD", "ZFC+AD", "ZFC+MM", "NBG", "MK", "KP", "Z2",
        "HOL", "CZF", "IZF", "NF", "NFU", "ML", "ZFC+I0", "ZFC+I1", "ZFC+I2"
    ]
    
    for system in systems:
        G.add_node(system, type="system")
    
    theorems = [
        "Con(PA)", "Con(ZFC)", "CH", "GCH", "PH", "GT", "AC", "MC", "PD", "AD", "MM", "PFA",
        "Fermat's Last Theorem", "Four Color Theorem", "Poincaré Conjecture", "Riemann Hypothesis",
        "P vs NP", "Twin Prime Conjecture", "Goldbach's Conjecture", "Collatz Conjecture",
        "ABC Conjecture", "Hodge Conjecture", "Navier-Stokes Existence and Smoothness",
        "Birch and Swinnerton-Dyer Conjecture", "Gödel's Incompleteness", "Continuum Hypothesis",
        "Axiom of Choice", "V=L", "BT", "SH", "Tarski's Undefinability", "Church-Turing Thesis",
        "Halting Problem", "Banach-Tarski Paradox", "Goodstein's Theorem", "Paris-Harrington Theorem"
    ]
    
    for theorem in theorems:
        G.add_node(theorem, type="theorem")
    
    # Define entailment relations
    entailments = [
        # System containment
        ("ZFC", "ZF", "Contains"),
        ("ZFC", "AC", "Contains"),
        ("ZF+DC", "ZF", "Contains"),
        ("ZFC+LC", "ZFC", "Contains"),
        ("ZFC+PD", "ZFC", "Contains"),
        ("ZFC+AD", "ZFC", "Contains"),
        ("ZFC+MM", "ZFC", "Contains"),
        ("PA+Con(PA)", "PA", "Contains"),
        ("PA+Con(PA)", "Con(PA)", "Contains"),
        ("PA2", "PA", "Contains"),
        ("ACA0", "RCA0", "Contains"),
        ("NBG", "ZFC", "Contains"),
        ("MK", "NBG", "Contains"),
        ("Z2", "PA2", "Contains"),
        ("ZFC+I0", "ZFC", "Contains"),
        ("ZFC+I1", "ZFC+I0", "Contains"),
        ("ZFC+I2", "ZFC+I1", "Contains"),
        
        # Provability relations
        ("ZFC", "Con(PA)", "Proves"),
        ("ZFC", "Fermat's Last Theorem", "Proves"),
        ("ZFC", "Four Color Theorem", "Proves"),
        ("ZFC", "Poincaré Conjecture", "Proves"),
        ("ZFC", "Hodge Conjecture", "Proves"),
        ("ZFC+LC", "Con(ZFC)", "Proves"),
        ("ZFC+LC", "MC", "Proves"),
        ("ZFC+PD", "PD", "Proves"),
        ("ZFC+PD", "BT", "Proves"),
        ("ZFC+AD", "AD", "Proves"),
        ("ZFC+MM", "MM", "Proves"),
        ("ZFC+MM", "PFA", "Proves"),
        ("PA+Con(PA)", "PH", "Proves"),
        ("PA+Con(PA)", "GT", "Proves"),
        ("PA2", "Goodstein's Theorem", "Proves"),
        ("PA2", "Paris-Harrington Theorem", "Proves"),
        ("ACA0", "PH", "Proves"),
        ("ACA0", "GT", "Proves"),
        ("Z2", "Banach-Tarski Paradox", "Proves"),
        ("PA", "Tarski's Undefinability", "Proves"),
        ("PA", "Halting Problem", "Proves"),
        
        # Independence relations
        ("ZFC", "CH", "Independence"),
        ("ZFC", "GCH", "Independence"),
        ("ZFC", "Con(ZFC)", "Independence"),
        ("ZFC", "SH", "Independence"),
        ("ZFC", "PFA", "Independence"),
        ("ZFC", "MM", "Independence"),
        ("ZFC", "Riemann Hypothesis", "Independence"),
        ("ZFC", "P vs NP", "Independence"),
        ("ZFC", "Twin Prime Conjecture", "Independence"),
        ("ZF", "AC", "Independence"),
        ("PA", "PH", "Independence"),
        ("PA", "GT", "Independence"),
        ("PA", "Con(PA)", "Independence"),
        ("PA", "Goodstein's Theorem", "Independence"),
        ("PA", "Paris-Harrington Theorem", "Independence"),
        
        # Contradictions
        ("ZFC+LC", "V=L", "Contradicts"),
        ("ZFC+AD", "AC", "Contradicts"),
        
        # Both independent and provable (in different systems)
        ("ZFC", "Continuum Hypothesis", "Independence"),
        ("ZFC+MM", "Continuum Hypothesis", "Contradicts"),
        ("ZF", "Axiom of Choice", "Independence"),
        ("ZFC", "Axiom of Choice", "Contains"),
        ("PA", "Gödel's Incompleteness", "Independence"),
        ("ZFC", "Gödel's Incompleteness", "Proves"),
        
        # Other relationships
        ("ZFC", "Goldbach's Conjecture", "Unknown"),
        ("ZFC", "Collatz Conjecture", "Unknown"),
        ("ZFC", "ABC Conjecture", "Unknown"),
        ("ZFC", "Navier-Stokes Existence and Smoothness", "Unknown"),
        ("ZFC", "Birch and Swinnerton-Dyer Conjecture", "Unknown"),
        ("ZFC", "Church-Turing Thesis", "Unknown")
    ]
    
    for source, target, relation in entailments:
        G.add_edge(source, target, relation=relation)
    
    # Save the graph to CSV
    filepath = os.path.join(OUTPUT_DIR, "expanded_independence_graph.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Relation"])
        for source, target, data in G.edges(data=True):
            writer.writerow([source, target, data["relation"]])
    
    print(f"Expanded independence entailment graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Saved to {filepath}")
    
    return G

def detect_bottlenecks(G):
    """
    Identify bottlenecks in the entailment structure.
    Bottlenecks are nodes with high betweenness centrality or cut vertices.
    """
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Find cut vertices (articulation points)
    undirected_G = G.to_undirected()
    try:
        cut_vertices = list(nx.articulation_points(undirected_G))
    except nx.NetworkXError:
        cut_vertices = []
        print("Warning: Could not compute articulation points")
    
    # Identify independence bridges
    independence_bridges = []
    for u, v, data in G.edges(data=True):
        if data['relation'] == "Independence":
            # Find nodes that can prove the target
            for node in G.nodes():
                if node != u and node != v:
                    if nx.has_path(G, node, v) and not nx.has_path(G, u, v):
                        independence_bridges.append((node, v))
    
    # Save results to CSV
    filepath = os.path.join(OUTPUT_DIR, "bottleneck_analysis.csv")
    with open(filepath, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Analysis Type", "Node", "Value", "Description"])
        
        # Write betweenness centrality
        for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
            writer.writerow(["Betweenness", node, f"{score:.4f}", f"Node type: {G.nodes[node].get('type', 'unknown')}"])
        
        # Write cut vertices
        for node in cut_vertices:
            writer.writerow(["Cut Vertex", node, "N/A", f"Node type: {G.nodes[node].get('type', 'unknown')}"])
        
        # Write independence bridges
        for source, target in independence_bridges:
            writer.writerow(["Independence Bridge", source, target, f"Source can prove target while original system cannot"])
    
    print(f"Bottleneck analysis saved to {filepath}")
    
    return {
        "betweenness": betweenness,
        "cut_vertices": cut_vertices,
        "independence_bridges": independence_bridges
    }

if __name__ == "__main__":
    print("Creating independence entailment graph...")
    G = create_independence_entailment_graph()
    
    print("\nAnalyzing bottlenecks...")
    analysis = detect_bottlenecks(G)
    
    print("\nBottleneck Analysis Results:")
    print("Top 5 nodes by betweenness centrality:")
    sorted_betweenness = sorted(analysis['betweenness'].items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in sorted_betweenness:
        print(f"  {node}: {score:.4f}")
    
    print("\nCut vertices (articulation points):")
    for node in analysis['cut_vertices']:
        print(f"  {node}")
    
    print("\nIndependence bridges:")
    for source, target in analysis['independence_bridges']:
        print(f"  {source} bridges to {target}")

