"""
Mathematical Logic Analysis Framework

This package provides tools for analyzing mathematical logic relationships,
specifically focusing on entailment relationships between logical statements.
"""

from mathlogic.core.entailment import LogicalStatement, EntailmentRelation, EntailmentCone
from mathlogic.core.statements import get_all_theorems, get_all_systems, get_all_relationships
from mathlogic.core.inference_rules import apply_modus_ponens, apply_reflexivity
from mathlogic.analysis.structural import (
    create_expanded_entailment_graph,
    compute_structural_metrics,
    classify_statements,
    analyze_neighborhood_structure,
    analyze_extended_neighborhood,
    generate_structural_analysis_report,
    visualize_centrality_distributions,
    visualize_network_structure,
    analyze_logical_strength,
    generate_enhanced_report,
    compute_advanced_metrics,
    analyze_independence_patterns,
    generate_independence_patterns_report
)
from mathlogic.analysis.dependencies import find_cut_nodes, find_required_axioms, save_axiom_dependencies
from mathlogic.analysis.open_problems import analyze_famous_open_problems, generate_report as generate_open_problems_report, visualize_problem_dependencies
from mathlogic.analysis.open_problems_analyzer import OpenProblemsAnalyzer # Expose the class itself
from mathlogic.utils.metrics import LogicalMetrics # Expose the class itself

__all__ = [
    'LogicalStatement',
    'EntailmentRelation',
    'EntailmentCone',
    'get_all_theorems',
    'get_all_systems',
    'get_all_relationships',
    'apply_modus_ponens',
    'apply_reflexivity',
    'create_expanded_entailment_graph',
    'compute_structural_metrics',
    'classify_statements',
    'analyze_neighborhood_structure',
    'analyze_extended_neighborhood',
    'generate_structural_analysis_report',
    'visualize_centrality_distributions',
    'visualize_network_structure',
    'analyze_logical_strength',
    'generate_enhanced_report',
    'compute_advanced_metrics',
    'analyze_independence_patterns',
    'generate_independence_patterns_report',
    'find_cut_nodes',
    'find_required_axioms',
    'save_axiom_dependencies',
    'analyze_famous_open_problems',
    'generate_open_problems_report',
    'visualize_problem_dependencies',
    'OpenProblemsAnalyzer',
    'LogicalMetrics'
]


