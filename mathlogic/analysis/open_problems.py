# New file to demonstrate enhanced analysis capabilities
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from mathlogic.analysis.structural import create_expanded_entailment_graph as create_graph_for_open_problems
from mathlogic.core.entailment import EntailmentCone, LogicalStatement, EntailmentRelation
from mathlogic.analysis.open_problems_analyzer import OpenProblemsAnalyzer
from mathlogic.utils.metrics import LogicalMetrics
# structural_analysis is now mathlogic.analysis.structural, so no need to import it separately here

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_graph_to_cone(G):
    """Convert a NetworkX graph to an EntailmentCone object."""
    cone = EntailmentCone()
    
    # Store LogicalStatement objects keyed by their node ID for easy lookup
    statements_map = {}
    for node_id, data in G.nodes(data=True):
        # The LogicalStatement constructor expects: content, statement_type, domain, complexity, statement_id, metadata
        # We are mapping NetworkX node attributes to LogicalStatement attributes
        content = node_id # Using node_id as content for simplicity, or data.get('content', node_id)
        statement_type = data.get('type', 'theorem') # 'system' or 'theorem'
        # is_axiom is not a direct parameter for LogicalStatement, but derived from type
        
        statement = LogicalStatement(
            content=content,
            statement_type=statement_type,
            statement_id=node_id # Use node_id as the unique identifier
        )
        cone.add_statement(statement)
        statements_map[node_id] = statement
    
    # Add relations
    for source_id, target_id, data in G.edges(data=True):
        source_statement = statements_map.get(source_id)
        target_statement = statements_map.get(target_id)
        
        if source_statement and target_statement:
            relation_type = data.get('relation', 'Implies')
            # EntailmentRelation constructor expects: source: LogicalStatement, target: LogicalStatement, relation_type: str
            relation = EntailmentRelation(source_statement, target_statement, relation_type)
            cone.add_relation(relation)
        else:
            print(f"Warning: Skipping relation from {source_id} to {target_id} as one or both statements not found.")
    
    return cone

def analyze_famous_open_problems(G):
    """Analyze famous open problems using the entailment cone."""
    # Convert graph to entailment cone
    cone = convert_graph_to_cone(G)
    
    # Create analyzer
    analyzer = OpenProblemsAnalyzer(cone)
    
    # Define open problems to analyze
    open_problems = [
        "CH",  # Continuum Hypothesis
        "GCH",  # Generalized Continuum Hypothesis
        "Con(ZFC)",  # Consistency of ZFC
        "PFA",  # Proper Forcing Axiom
        "Suslin Hypothesis"  # Suslin Hypothesis
    ]
    
    # Analyze each problem
    results = {}
    for problem in open_problems:
        if problem in G.nodes():
            print(f"\nAnalyzing {problem}...")
            analysis = analyzer.analyze_open_problem(problem)
            results[problem] = analysis
            
            # Print key findings
            print(f"  Logical strength: {analysis.get('logical_strength', 0):.4f}")
            print(f"  Bottleneck centrality: {analysis.get('bottleneck_centrality', 0):.4f}")
            
            # Print independence likelihood for top systems
            print("  Independence likelihood by system:")
            independence = analysis.get('independence_likelihood', {})
            sorted_independence = sorted(independence.items(), key=lambda x: x[1], reverse=True)[:3]
            for system, likelihood in sorted_independence:
                print(f"    {system}: {likelihood:.4f}")
            
            # Print potential resolution axioms
            print("  Potential resolution axioms:")
            resolution = analysis.get('potential_resolution_axioms', {})
            for system, axioms in list(resolution.items())[:2]:
                print(f"    For {system}: {', '.join(axioms[:3])}")
        else:
            print(f"\nSkipping {problem} - not found in graph")
    
    return results

def generate_report(results):
    """Generate a comprehensive report of the analysis results."""
    report_path = os.path.join(OUTPUT_DIR, "open_problems_analysis.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Analysis of Open Problems in Mathematical Logic\n\n")
        f.write("This report presents an analysis of several open problems using entailment cone methodology.\n\n")
        
        for problem, analysis in results.items():
            f.write(f"## {problem}\n\n")
            
            # Logical strength
            strength = analysis.get('logical_strength', 0)
            f.write(f"**Logical Strength:** {strength:.4f}\n\n")
            
            # Bottleneck centrality
            centrality = analysis.get('bottleneck_centrality', 0)
            f.write(f"**Bottleneck Centrality:** {centrality:.4f}\n\n")
            
            # Independence likelihood
            f.write("### Independence Likelihood\n\n")
            independence = analysis.get('independence_likelihood', {})
            f.write("| System | Likelihood |\n")
            f.write("|--------|------------|\n")
            sorted_independence = sorted(independence.items(), key=lambda x: x[1], reverse=True)
            for system, likelihood in sorted_independence:
                f.write(f"| {system} | {likelihood:.4f} |\n")
            f.write("\n")
            
            # Resolution axioms
            f.write("### Potential Resolution Axioms\n\n")
            resolution = analysis.get('potential_resolution_axioms', {})
            for system, axioms in resolution.items():
                f.write(f"**For {system}:**\n\n")
                if axioms:
                    for axiom in axioms:
                        f.write(f"- {axiom}\n")
                else:
                    f.write("- No additional axioms needed\n")
                f.write("\n")
            
            # Related theorems
            f.write("### Related Theorems\n\n")
            related = analysis.get('related_theorems', [])
            if related:
                for theorem in related:
                    f.write(f"- {theorem}\n")
            else:
                f.write("- No closely related theorems identified\n")
            f.write("\n")
    
    print(f"Report generated: {report_path}")
    return report_path

def visualize_problem_dependencies(G, problem, output_dir=OUTPUT_DIR):
    """Create a visualization of a problem's dependencies."""
    # Create a subgraph containing the problem and its neighborhood
    # Original approach: just immediate neighbors
    # neighbors = list(G.predecessors(problem)) + list(G.successors(problem))
    # subgraph = G.subgraph([problem] + neighbors)
    
    # Enhanced approach: get 2-hop neighborhood for more context
    one_hop = set(G.predecessors(problem)).union(set(G.successors(problem)))
    two_hop = set()
    for node in one_hop:
        two_hop.update(G.predecessors(node))
        two_hop.update(G.successors(node))
    
    # Create subgraph with all relevant nodes
    nodes = {problem}.union(one_hop).union(two_hop)
    subgraph = G.subgraph(nodes)
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Draw nodes with different colors based on type
    system_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'system']
    theorem_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'theorem']
    
    # Highlight the problem node
    nx.draw_networkx_nodes(subgraph, pos, 
                          nodelist=[problem],
                          node_color='red',
                          node_size=800)
    
    # Draw other nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          nodelist=[n for n in system_nodes if n != problem],
                          node_color='skyblue',
                          node_size=500)
    nx.draw_networkx_nodes(subgraph, pos, 
                          nodelist=[n for n in theorem_nodes if n != problem],
                          node_color='lightgreen',
                          node_size=500)
    
    # Draw edges with different styles based on relation
    # Get edges by relation type
    proves_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('relation') == 'Proves']
    implies_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('relation') == 'Implies']
    independent_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('relation') == 'Independent']
    contains_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('relation') == 'Contains']
    contradicts_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('relation') == 'Contradicts']
    
    # Draw edges with different styles
    nx.draw_networkx_edges(subgraph, pos, edgelist=proves_edges, edge_color='blue', width=2)
    nx.draw_networkx_edges(subgraph, pos, edgelist=implies_edges, edge_color='green', width=1.5)
    nx.draw_networkx_edges(subgraph, pos, edgelist=independent_edges, edge_color='red', style='dashed', width=1.5)
    nx.draw_networkx_edges(subgraph, pos, edgelist=contains_edges, edge_color='purple', width=1.5)
    nx.draw_networkx_edges(subgraph, pos, edgelist=contradicts_edges, edge_color='orange', style='dotted', width=2)
    
    # Add labels
    nx.draw_networkx_labels(subgraph, pos)
    
    # Add a legend
    plt.legend([
        plt.Line2D([0], [0], color='red', marker='o', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='skyblue', marker='o', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='lightgreen', marker='o', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='blue', linewidth=2),
        plt.Line2D([0], [0], color='green', linewidth=1.5),
        plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='dashed'),
        plt.Line2D([0], [0], color='purple', linewidth=1.5),
        plt.Line2D([0], [0], color='orange', linewidth=2, linestyle='dotted')
    ], [
        'Target Problem',
        'Formal Systems',
        'Theorems',
        'Proves',
        'Implies',
        'Independent',
        'Contains',
        'Contradicts'
    ], loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the figure
    plt.title(f"Dependency Graph for {problem}")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{problem}_dependencies.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def main():
    """Main function to run the analysis."""
    print("Creating expanded entailment graph for open problems analysis...")
    G = create_graph_for_open_problems()
    
    print("\nAnalyzing famous open problems...")
    results = analyze_famous_open_problems(G)
    
    print("\nGenerating comprehensive report...")
    report_path = generate_report(results)
    
    print("\nCreating visualizations for each problem...")
    for problem in results.keys():
        viz_path = visualize_problem_dependencies(G, problem)
        print(f"  Visualization for {problem} saved to {viz_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
