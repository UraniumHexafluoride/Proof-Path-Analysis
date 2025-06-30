from mathlogic.core.entailment import EntailmentCone, LogicalStatement, EntailmentRelation
# from relationship_discovery import RelationshipDiscovery # Commented out for now, needs review
# from structural_analysis import StructuralAnalysis # Replaced by direct calls to structural functions
from mathlogic.utils.metrics import LogicalMetrics
import networkx as nx
from typing import Dict, List, Tuple, Set, Any, Optional
import os
import logging
import json

# Import specific structural analysis functions needed
from mathlogic.analysis.structural import analyze_independence_patterns, compute_structural_metrics, classify_statements
from mathlogic.analysis.dependencies import find_cut_nodes # Assuming identify_bottlenecks might relate to cut nodes

class OpenProblemsAnalyzer:
    """Analyzes open problems using entailment cone methodology."""
    
    def __init__(self, cone: EntailmentCone):
        """
        Initialize the analyzer.
        
        Args:
            cone: EntailmentCone object containing the logical statements and relations
        """
        self.cone = cone
        self.G = cone.graph
        # self.relationship_discovery = RelationshipDiscovery(cone) # Needs to be re-evaluated
        # self.structural_analysis = StructuralAnalysis(cone) # Replaced
        self.logical_metrics = LogicalMetrics(cone)
        
        # Pre-compute metrics needed for analysis
        self.metrics = compute_structural_metrics(self.G)
        self.classifications = classify_statements(self.G)
        # Assuming analyze_independence_patterns can provide the 'independence_structure'
        # This might need a more specific function if 'analyze_independence_structure'
        # is not directly available or has a different signature.
        # For now, we'll use a simplified approach or assume it's part of a broader 'patterns' result.
        # self.independence_patterns = analyze_independence_patterns(self.G, self.classifications, self.metrics, {})
        
        self.metrics_cache = {}
    
    def analyze_problems(self, G: nx.DiGraph, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze open problems in the graph.
        
        Args:
            G: NetworkX DiGraph representing the entailment graph
            metrics: Dictionary of precomputed graph metrics
            
        Returns:
            Dictionary containing analysis results for each open problem
        """
        results = {}
        
        # Define open problems to analyze
        open_problems = [
            "Continuum Hypothesis",
            "Twin Prime Conjecture",
            "Riemann Hypothesis",
            "Goldbach's Conjecture",
            "P vs NP"
        ]
        
        for problem in open_problems:
            if problem in G.nodes():
                logging.info(f"Analyzing {problem}...")
                analysis = self.analyze_open_problem(problem)
                results[problem] = analysis
                
                # Log key findings
                self._log_analysis_results(problem, analysis)
            else:
                logging.warning(f"Problem {problem} not found in graph")
        
        return results
    
    def analyze_open_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analyze a single open problem.
        
        Args:
            problem: Name of the problem to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Get the logical statement from the cone
        statement = self.cone.get_statement(problem)
        if not statement:
            logging.warning(f"Problem {problem} not found in entailment cone")
            return {}
        
        # Compute logical strength
        strength = self._compute_logical_strength(statement)
        
        # Compute bottleneck centrality
        centrality = self._compute_bottleneck_centrality(statement)
        
        # Analyze independence likelihood
        independence = self._analyze_independence_likelihood(statement)
        
        # Find potential resolution axioms
        resolution = self._find_resolution_axioms(statement)
        
        # Find related theorems
        related = self._find_related_theorems(statement)
        
        return {
            'logical_strength': strength,
            'bottleneck_centrality': centrality,
            'independence_likelihood': independence,
            'potential_resolution_axioms': resolution,
            'related_theorems': related
        }
    
    def _compute_logical_strength(self, statement) -> float:
        """Compute logical strength of a statement."""
        # Implementation depends on specific metrics
        return 0.0
    
    def _compute_bottleneck_centrality(self, statement) -> float:
        """Compute bottleneck centrality of a statement."""
        # Implementation depends on graph structure
        return 0.0
    
    def _analyze_independence_likelihood(self, statement) -> Dict[str, float]:
        """Analyze likelihood of independence from various systems."""
        return {}
    
    def _find_resolution_axioms(self, statement) -> Dict[str, List[str]]:
        """Find potential axioms that could resolve the problem."""
        return {}
    
    def _find_related_theorems(self, statement) -> List[str]:
        """Find theorems related to the problem."""
        return []
    
    def _log_analysis_results(self, problem: str, analysis: Dict[str, Any]):
        """Log key findings from the analysis."""
        logging.info(f"\nAnalysis results for {problem}:")
        logging.info(f"  Logical strength: {analysis.get('logical_strength', 0):.4f}")
        logging.info(f"  Bottleneck centrality: {analysis.get('bottleneck_centrality', 0):.4f}")
        
        independence = analysis.get('independence_likelihood', {})
        if independence:
            logging.info("  Independence likelihood by system:")
            for system, likelihood in sorted(independence.items(), key=lambda x: x[1], reverse=True)[:3]:
                logging.info(f"    {system}: {likelihood:.4f}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save analysis results to a file.
        
        Args:
            results: Analysis results to save
            output_file: Path to save the results
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved analysis results to {output_file}")
        except Exception as e:
            logging.error(f"Error saving analysis results: {str(e)}")
    
    def predict_independence_likelihood(self, statement: str, system: str) -> float:
        """
        Predict the likelihood that a statement is independent of a formal system.
        Uses structural patterns from known independence results.
        """
        # Check if statement and system exist in the cone
        if statement not in self.cone.statements or system not in self.G.nodes():
            return 0.0
        
        # To get 'system_independents', we need a way to query known independent statements
        # related to a system. This logic was previously in StructuralAnalysis.
        # For now, we'll simulate it based on classifications.
        system_independents = [
            node for node, cls in self.classifications.items()
            if cls == 'independent' and nx.has_path(self.G, system, node) # Simplified check
        ]
        
        if not system_independents:
            return 0.1  # Low baseline if no known independence results
        
        # Compute similarity to known independent statements
        similarities = []
        for independent in system_independents:
            # Compute structural similarity based on shared neighbors
            statement_neighbors = set(self.G.successors(statement)).union(set(self.G.predecessors(statement)))
            independent_neighbors = set(self.G.successors(independent)).union(set(self.G.predecessors(independent)))
            
            # Jaccard similarity of neighborhoods
            if statement_neighbors or independent_neighbors:
                similarity = len(statement_neighbors.intersection(independent_neighbors)) / \
                             len(statement_neighbors.union(independent_neighbors))
                similarities.append(similarity)
        
        # Average similarity to known independent statements
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Check if there's a direct path from system to statement
        has_path = nx.has_path(self.G, system, statement)
        
        # Combine factors to estimate independence likelihood
        if has_path:
            return 0.1 * avg_similarity  # Low likelihood if provable
        else:
            return 0.5 + (0.5 * avg_similarity)  # Higher baseline if no path exists
    
    def suggest_axiom_extensions(self, system: str, target_statement: str) -> List[str]:
        """
        Suggest minimal axiom extensions to a formal system that would allow
        it to prove a target statement.
        """
        if system not in self.G.nodes() or target_statement not in self.G.nodes():
            return []
        
        # Check if system already proves target
        if nx.has_path(self.G, system, target_statement):
            return []  # No extension needed
        
        # Find all statements that can prove the target
        provers = []
        for node in self.G.nodes():
            if node != system and node != target_statement:
                if nx.has_path(self.G, node, target_statement):
                    provers.append(node)
        
        # Rank potential axioms by efficiency (using a placeholder for now)
        # The original compute_axiom_efficiency was in LogicalMetrics, but its implementation
        # is not directly available here. For now, we'll use a dummy ranking.
        # A proper implementation would involve calling a method from self.logical_metrics
        # that computes this based on the graph.
        
        # Dummy axiom efficiency for demonstration
        dummy_axiom_efficiency = {p: self.G.out_degree(p) for p in provers} # More outgoing edges = more "efficient"
        
        ranked_provers = sorted(provers,
                               key=lambda p: dummy_axiom_efficiency.get(p, 0),
                               reverse=True)
        
        return ranked_provers[:5]  # Return top 5 most efficient axioms
    
    def identify_potential_theorems(self, statement: str) -> List[Tuple[str, float]]:
        """
        Identify statements that might be provable from a given statement,
        but where the proof is not yet known.
        """
        if statement not in self.G.nodes():
            return []
        
        potential_theorems = []
        
        # Get statements directly provable from this one
        known_theorems = set(self.G.successors(statement))
        
        # Get all statements
        all_statements = set(self.G.nodes())
        
        # Remove the statement itself and known theorems
        candidates = all_statements - known_theorems - {statement}
        
        # For each candidate, estimate likelihood it's a theorem
        for candidate in candidates:
            # Skip if there's already a path (indirect proof)
            if nx.has_path(self.G, statement, candidate):
                continue
                
            # Compute features that suggest a proof might exist
            
            # 1. Logical distance to known theorems
            min_distance = float('inf')
            for theorem in known_theorems:
                try:
                    distance = nx.shortest_path_length(self.G, theorem, candidate)
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    pass
            
            # 2. Shared successors with statement
            statement_successors = set(nx.descendants(self.G, statement))
            candidate_successors = set(nx.descendants(self.G, candidate))
            shared_successors = len(statement_successors.intersection(candidate_successors))
            
            # 3. Shared predecessors with known theorems
            statement_predecessors = set(nx.ancestors(self.G, statement))
            candidate_predecessors = set(nx.ancestors(self.G, candidate))
            shared_predecessors = len(statement_predecessors.intersection(candidate_predecessors))
            
            # Combine features to estimate theorem likelihood
            if min_distance < float('inf'):
                distance_factor = 1.0 / (1.0 + min_distance)
            else:
                distance_factor = 0.0
                
            successor_factor = shared_successors / (len(statement_successors) + 1)
            predecessor_factor = shared_predecessors / (len(statement_predecessors) + 1)
            
            # Weighted combination
            likelihood = (0.5 * distance_factor +
                         0.3 * successor_factor +
                         0.2 * predecessor_factor)
            
            if likelihood > 0.1:  # Only include reasonably likely candidates
                potential_theorems.append((candidate, likelihood))
        
        # Sort by likelihood
        return sorted(potential_theorems, key=lambda x: x[1], reverse=True)

# The main function for demonstration purposes should be in run_analysis.py
# This block is for standalone testing of OpenProblemsAnalyzer if needed.
if __name__ == "__main__":
    print("This script is primarily a module for the mathlogic package.")
    print("For a full demonstration, run 'python run_analysis.py'.")
    
    # Example usage for testing this module directly
    # from mathlogic.core.statements import get_all_theorems, get_all_systems, get_all_relationships
    # from mathlogic.analysis.structural import create_expanded_entailment_graph
    
    # G_demo = create_expanded_entailment_graph()
    # cone_demo = EntailmentCone()
    # for node, data in G_demo.nodes(data=True):
    #     cone_demo.add_statement(LogicalStatement(node, "", data.get('type', 'theorem'), data.get('type') == 'system'))
    # for u, v, data in G_demo.edges(data=True):
    #     cone_demo.add_relation(EntailmentRelation(u, v, data.get('relation', 'Implies')))
    
    # analyzer_demo = OpenProblemsAnalyzer(cone_demo)
    # analysis_result = analyzer_demo.analyze_open_problem("Continuum Hypothesis")
    # print("\nDemo Analysis of Continuum Hypothesis:")
    # print(analysis_result)
