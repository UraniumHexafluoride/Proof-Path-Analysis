"""
Validation script to test independence likelihood predictions against known results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from expanded_entailment_data import create_expanded_entailment_graph
from open_problems_analyzer import OpenProblemsAnalyzer
from entailment_theory import EntailmentCone, LogicalStatement, EntailmentRelation

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_graph_to_cone(G):
    """Convert a NetworkX graph to an EntailmentCone object."""
    cone = EntailmentCone()
    
    # Add statements
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'theorem')
        description = data.get('description', '')
        is_axiom = node_type == 'system'
        statement = LogicalStatement(node, description, node_type, is_axiom)
        cone.add_statement(statement)
    
    # Add relations
    for source, target, data in G.edges(data=True):
        relation = data.get('relation', 'Implies')
        cone.add_relation(EntailmentRelation(source, target, relation))
    
    return cone

def get_known_independence_results():
    """Return a dictionary of known independence results."""
    return {
        # Format: (statement, system): True if independent, False if not independent
        ("CH", "ZFC"): True,       # Continuum Hypothesis is independent of ZFC
        ("GCH", "ZFC"): True,      # Generalized Continuum Hypothesis is independent of ZFC
        ("AC", "ZF"): True,        # Axiom of Choice is independent of ZF
        ("PH", "PA"): True,        # Paris-Harrington is independent of PA
        ("GT", "PA"): True,        # Goodstein's Theorem is independent of PA
        ("Con(PA)", "PA"): True,   # Consistency of PA is independent of PA (Gödel)
        ("Con(ZFC)", "ZFC"): True, # Consistency of ZFC is independent of ZFC (Gödel)
        ("SH", "ZFC"): True,       # Suslin Hypothesis is independent of ZFC
        
        # Known non-independence results
        ("PH", "ACA0"): False,     # Paris-Harrington is provable in ACA0
        ("GT", "ACA0"): False,     # Goodstein's Theorem is provable in ACA0
        ("AC", "ZFC"): False,      # AC is part of ZFC
        ("PH", "PA2"): False,      # Paris-Harrington is provable in PA2
        ("IVT", "RCA0"): False,    # Intermediate Value Theorem is provable in RCA0
        ("FTA", "ZFC"): False,     # Fundamental Theorem of Algebra is provable in ZFC
    }

def validate_independence_predictions(G):
    """Validate independence likelihood predictions against known results."""
    # Convert graph to entailment cone
    cone = convert_graph_to_cone(G)
    
    # Create analyzer
    analyzer = OpenProblemsAnalyzer(cone)
    
    # Get known independence results
    known_results = get_known_independence_results()
    
    # Validate predictions
    results = []
    for (statement, system), is_independent in known_results.items():
        if statement in G.nodes() and system in G.nodes():
            # Get prediction
            analysis = analyzer.analyze_open_problem(statement)
            independence_likelihood = analysis.get('independence_likelihood', {}).get(system, 0)
            
            # Determine prediction (threshold at 0.5)
            predicted_independent = independence_likelihood > 0.5
            
            # Compare with known result
            correct = (predicted_independent == is_independent)
            
            results.append({
                'statement': statement,
                'system': system,
                'known_independent': is_independent,
                'predicted_likelihood': independence_likelihood,
                'predicted_independent': predicted_independent,
                'correct': correct
            })
    
    return results

def generate_validation_report(validation_results):
    """Generate a report of validation results."""
    report_path = os.path.join(OUTPUT_DIR, "validation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Independence Prediction Validation Report\n\n")
        f.write("This report compares our model's independence predictions with known results.\n\n")
        
        # Overall accuracy
        correct_count = sum(1 for r in validation_results if r['correct'])
        total_count = len(validation_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        f.write(f"## Overall Accuracy\n\n")
        f.write(f"**Accuracy:** {accuracy:.2%} ({correct_count}/{total_count})\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        f.write("| Statement | System | Known Independent | Predicted Likelihood | Predicted Independent | Correct |\n")
        f.write("|-----------|--------|-------------------|----------------------|-----------------------|--------|\n")
        
        for result in validation_results:
            f.write(f"| {result['statement']} | {result['system']} | ")
            f.write(f"{'Yes' if result['known_independent'] else 'No'} | ")
            f.write(f"{result['predicted_likelihood']:.4f} | ")
            f.write(f"{'Yes' if result['predicted_independent'] else 'No'} | ")
            f.write(f"{'✓' if result['correct'] else '✗'} |\n")
        
        # Analysis of errors
        f.write("\n## Error Analysis\n\n")
        
        errors = [r for r in validation_results if not r['correct']]
        if errors:
            f.write("The model made incorrect predictions for the following cases:\n\n")
            for error in errors:
                f.write(f"- **{error['statement']} from {error['system']}**: ")
                f.write(f"Predicted {'independent' if error['predicted_independent'] else 'provable'} ")
                f.write(f"(likelihood: {error['predicted_likelihood']:.4f}), ")
                f.write(f"but actually {'independent' if error['known_independent'] else 'provable'}.\n")
        else:
            f.write("The model made no errors on the validation set.\n")
        
        # Confusion matrix
        f.write("\n## Confusion Matrix\n\n")
        
        # Calculate confusion matrix values
        true_positive = sum(1 for r in validation_results if r['known_independent'] and r['predicted_independent'])
        false_positive = sum(1 for r in validation_results if not r['known_independent'] and r['predicted_independent'])
        true_negative = sum(1 for r in validation_results if not r['known_independent'] and not r['predicted_independent'])
        false_negative = sum(1 for r in validation_results if r['known_independent'] and not r['predicted_independent'])
        
        f.write("```\n")
        f.write("                  | Predicted Independent | Predicted Provable |\n")
        f.write("------------------|----------------------|--------------------|\n")
        f.write(f"Actually Independent | {true_positive:20} | {false_negative:19} |\n")
        f.write(f"Actually Provable    | {false_positive:20} | {true_negative:19} |\n")
        f.write("```\n\n")
        
        # Calculate metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Precision:** {precision:.4f}\n")
        f.write(f"- **Recall:** {recall:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if accuracy < 0.7:
            f.write("The model's accuracy is below 70%, suggesting significant room for improvement. Recommendations:\n\n")
            f.write("1. Expand the entailment graph with more known independence results\n")
            f.write("2. Refine the structural metrics used for prediction\n")
            f.write("3. Consider using machine learning to improve prediction accuracy\n")
        elif accuracy < 0.9:
            f.write("The model's accuracy is good but could be improved. Recommendations:\n\n")
            f.write("1. Focus on the specific error cases identified above\n")
            f.write("2. Add more fine-grained structural metrics\n")
            f.write("3. Consider the specific characteristics of misclassified statements\n")
        else:
            f.write("The model's accuracy is excellent. Recommendations:\n\n")
            f.write("1. Apply the model to open problems with unknown independence status\n")
            f.write("2. Validate the model on a larger set of known results\n")
            f.write("3. Consider publishing the approach as a predictive tool for independence\n")
    
    return report_path

def visualize_validation_results(validation_results):
    """Create visualizations of the validation results."""
    # Create DataFrame
    df = pd.DataFrame(validation_results)
    
    # Create ROC-like plot
    plt.figure(figsize=(10, 8))
    
    # Sort by predicted likelihood
    df_sorted = df.sort_values('predicted_likelihood', ascending=False)
    
    # Calculate cumulative true positives and false positives
    df_sorted['cum_tp'] = (df_sorted['known_independent'] == True).cumsum()
    df_sorted['cum_fp'] = (df_sorted['known_independent'] == False).cumsum()
    
    # Normalize
    total_positive = df_sorted['known_independent'].sum()
    total_negative = len(df_sorted) - total_positive
    
    if total_positive > 0 and total_negative > 0:
        df_sorted['tpr'] = df_sorted['cum_tp'] / total_positive
        df_sorted['fpr'] = df_sorted['cum_fp'] / total_negative
        
        # Plot ROC-like curve
        plt.plot(df_sorted['fpr'], df_sorted['tpr'], 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-like Curve for Independence Prediction')
        plt.grid(True)
        
        # Save figure
        roc_path = os.path.join(OUTPUT_DIR, "validation_roc.png")
        plt.savefig(roc_path)
        plt.close()
    
    # Create likelihood distribution plot
    plt.figure(figsize=(10, 8))
    
    # Create bins for likelihood
    bins = np.linspace(0, 1, 11)
    
    # Plot histograms
    plt.hist(df[df['known_independent']]['predicted_likelihood'], bins=bins, alpha=0.5, label='Actually Independent')
    plt.hist(df[~df['known_independent']]['predicted_likelihood'], bins=bins, alpha=0.5, label='Actually Provable')
    
    plt.xlabel('Predicted Independence Likelihood')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Likelihoods')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    dist_path = os.path.join(OUTPUT_DIR, "validation_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    
    return [roc_path, dist_path]

def main():
    """Main function to run the validation."""
    print("Creating expanded entailment graph...")
    G = create_expanded_entailment_graph()
    
    print("\nValidating independence predictions...")
    validation_results = validate_independence_predictions(G)
    
    print("\nGenerating validation report...")
    report_path = generate_validation_report(validation_results)
    
    print("\nCreating visualizations...")
    viz_paths = visualize_validation_results(validation_results)
    
    print(f"\nValidation complete! Report saved to {report_path}")
    for path in viz_paths:
        print(f"Visualization saved to {path}")
    
    # Print summary
    correct_count = sum(1 for r in validation_results if r['correct'])
    total_count = len(validation_results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\nOverall accuracy: {accuracy:.2%} ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()


