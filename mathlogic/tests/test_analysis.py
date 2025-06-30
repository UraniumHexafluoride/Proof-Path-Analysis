"""
Tests for analysis functionality of the mathlogic package.
"""

import pytest
import networkx as nx
from mathlogic.analysis.structural import StructuralAnalyzer, AnalysisResult
from mathlogic.core.entailment import EntailmentCone, LogicalStatement, EntailmentRelation

@pytest.fixture
def complex_cone(basic_cone, basic_statements):
    """Fixture providing a more complex entailment cone for testing analysis."""
    # Add more statements
    new_statements = {
        'gch': LogicalStatement("GCH", "Generalized Continuum Hypothesis", "Set Theory", False),
        'ph': LogicalStatement("PH", "Paris-Harrington Theorem", "Number Theory", False),
        'gt': LogicalStatement("GT", "Goodstein Theorem", "Number Theory", False),
        'aca0': LogicalStatement("ACA0", "Arithmetical Comprehension Axiom", "Number Theory", True)
    }
    
    for stmt in new_statements.values():
        basic_cone.add_statement(stmt)
    
    # Add more complex relationships
    relations = [
        EntailmentRelation(basic_statements['zfc'], new_statements['gch'], "Independence"),
        EntailmentRelation(basic_statements['pa'], new_statements['ph'], "Independence"),
        EntailmentRelation(basic_statements['pa'], new_statements['gt'], "Independence"),
        EntailmentRelation(new_statements['aca0'], new_statements['ph'], "Proves"),
        EntailmentRelation(new_statements['aca0'], new_statements['gt'], "Proves"),
        EntailmentRelation(basic_statements['zfc'], basic_statements['pa'], "Contains")
    ]
    
    for rel in relations:
        basic_cone.add_relation(rel)
    
    return basic_cone

def test_analyzer_creation(complex_cone):
    """Test creation of structural analyzer."""
    analyzer = StructuralAnalyzer(complex_cone)
    assert analyzer.cone == complex_cone
    assert analyzer.graph == complex_cone.graph
    assert isinstance(analyzer._cached_metrics, dict)

def test_statement_classification(complex_cone):
    """Test classification of statements."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    classifications = result.classifications
    assert isinstance(classifications, dict)
    
    # Check specific classifications
    assert classifications['PH'] == 'both'  # Both independent and proved
    assert classifications['GCH'] == 'independent'  # Only independent
    assert classifications['AC'] == 'provable'  # Proved via Contains

def test_pattern_identification(complex_cone):
    """Test identification of structural patterns."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    patterns = result.patterns
    assert isinstance(patterns, dict)
    assert 'independence_clusters' in patterns
    assert 'proof_bottlenecks' in patterns
    assert 'bridge_statements' in patterns
    assert 'central_theorems' in patterns

def test_metric_computation(complex_cone):
    """Test computation of structural metrics."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    metrics = result.metrics
    assert 'centrality' in metrics
    assert 'independence' in metrics
    assert 'clustering' in metrics
    
    # Check specific metrics
    centrality = metrics['centrality']
    for node in complex_cone.graph.nodes():
        assert node in centrality
        assert 'degree' in centrality[node]
        assert 'betweenness' in centrality[node]
        assert 'closeness' in centrality[node]
        assert 'eigenvector' in centrality[node]

def test_independence_clusters(complex_cone):
    """Test identification of independence clusters."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    clusters = result._find_independence_clusters()
    assert isinstance(clusters, list)
    assert all(isinstance(cluster, set) for cluster in clusters)
    
    # Check if related independent statements are clustered
    ch_cluster = None
    for cluster in clusters:
        if 'CH' in cluster:
            ch_cluster = cluster
            break
    
    assert ch_cluster is not None
    assert 'GCH' in ch_cluster  # GCH should be in same cluster as CH

def test_bottleneck_identification(complex_cone):
    """Test identification of bottleneck statements."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    bottlenecks = result._identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    assert 'ZFC' in bottlenecks  # ZFC should be a bottleneck as it's central

def test_recommendation_generation(complex_cone):
    """Test generation of analysis recommendations."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    recommendations = result.recommendations
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)

def test_analysis_result_creation(complex_cone):
    """Test creation and structure of analysis results."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    assert isinstance(result, AnalysisResult)
    assert hasattr(result, 'metrics')
    assert hasattr(result, 'classifications')
    assert hasattr(result, 'patterns')
    assert hasattr(result, 'recommendations')

def test_cached_metrics(complex_cone):
    """Test caching of computed metrics."""
    analyzer = StructuralAnalyzer(complex_cone)
    
    # First computation should cache results
    result1 = analyzer.analyze_structure()
    cached_metrics = analyzer._cached_metrics
    
    # Second computation should use cache
    result2 = analyzer.analyze_structure()
    assert analyzer._cached_metrics == cached_metrics
    
    # Results should be identical
    assert result1.metrics == result2.metrics

def test_invalid_graph_handling():
    """Test handling of invalid or empty graphs."""
    empty_cone = EntailmentCone()
    analyzer = StructuralAnalyzer(empty_cone)
    result = analyzer.analyze_structure()
    
    # Should handle empty graph gracefully
    assert len(result.classifications) == 0
    assert len(result.patterns['independence_clusters']) == 0
    assert len(result.patterns['proof_bottlenecks']) == 0

@pytest.mark.parametrize("metric_name", [
    'degree_centrality',
    'betweenness_centrality',
    'closeness_centrality',
    'pagerank'
])
def test_specific_metrics(complex_cone, metric_name):
    """Test computation of specific metrics."""
    analyzer = StructuralAnalyzer(complex_cone)
    result = analyzer.analyze_structure()
    
    # Check if metric exists for all nodes
    for node in complex_cone.graph.nodes():
        assert metric_name in result.metrics['centrality'][node]
        assert isinstance(result.metrics['centrality'][node][metric_name], float)
        assert 0 <= result.metrics['centrality'][node][metric_name] <= 1

