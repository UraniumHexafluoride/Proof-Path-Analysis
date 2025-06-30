from mathlogic.core.entailment import EntailmentCone
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set

class LogicalMetrics:
    """Metrics for quantifying and comparing logical strength."""
    
    def __init__(self, cone: EntailmentCone):
        self.cone = cone
        self.G = cone.graph
    
    def compute_proof_power(self) -> Dict[str, float]:
        """
        Compute the 'proof power' of each statement.
        Proof power measures how many other statements can be derived.
        """
        proof_power = {}
        
        for node in self.G.nodes():
            # Count reachable nodes (excluding self)
            reachable = nx.descendants(self.G, node)
            if node in reachable:
                reachable.remove(node)
            
            # Normalize by total number of statements
            total_statements = len(self.G.nodes()) - 1  # Exclude self
            if total_statements > 0:
                proof_power[node] = len(reachable) / total_statements
            else:
                proof_power[node] = 0
                
        return proof_power
    
    def compute_axiom_efficiency(self) -> Dict[str, float]:
        """
        Compute the efficiency of axioms.
        Efficiency = (number of theorems proved) / (logical complexity of axiom)
        """
        efficiency = {}
        
        for node, data in self.G.nodes(data=True):
            if data.get('is_axiom', False):
                # Count theorems proved
                theorems_proved = len(nx.descendants(self.G, node))
                
                # Estimate logical complexity (using string length as proxy)
                statement = self.cone.statements.get(node)
                if statement:
                    complexity = len(statement.description)
                    if complexity > 0:
                        efficiency[node] = theorems_proved / complexity
                    else:
                        efficiency[node] = 0
        
        return efficiency
    
    def compute_independence_resistance(self) -> Dict[str, float]:
        """
        Compute 'independence resistance' of formal systems.
        Measures how resistant a system is to independence results.
        """
        resistance = {}
        
        # Group statements by formal system
        systems = {}
        for node, data in self.G.nodes(data=True):
            system = data.get('formal_system')
            if system:
                if system not in systems:
                    systems[system] = []
                systems[system].append(node)
        
        # For each system, compute independence resistance
        for system, statements in systems.items():
            independence_count = 0
            total_relations = 0
            
            for statement in statements:
                for _, target, data in self.G.out_edges(statement, data=True):
                    total_relations += 1
                    if data.get('relation_type') == 'Independence':
                        independence_count += 1
            
            if total_relations > 0:
                resistance[system] = 1 - (independence_count / total_relations)
            else:
                resistance[system] = 1.0  # Default to full resistance if no relations
                
        return resistance
    
    def compute_logical_strength_index(self) -> Dict[str, float]:
        """
        Compute a composite index of logical strength.
        Combines multiple metrics into a single strength score.
        """
        # Get component metrics
        proof_power = self.compute_proof_power()
        
        # Get centrality metrics
        centrality = nx.eigenvector_centrality(self.G, max_iter=1000)
        
        # Compute logical strength as weighted combination
        strength_index = {}
        for node in self.G.nodes():
            power = proof_power.get(node, 0)
            central = centrality.get(node, 0)
            
            # Weighted combination (can be adjusted)
            strength_index[node] = 0.7 * power + 0.3 * central
            
        return strength_index
    
    def compare_formal_systems(self) -> Dict[Tuple[str, str], float]:
        """
        Compare the relative strength of formal systems.
        Returns a dictionary of system pairs with their strength ratios.
        """
        # Group statements by formal system
        systems = {}
        for node, data in self.G.nodes(data=True):
            system = data.get('formal_system')
            if system:
                if system not in systems:
                    systems[system] = []
                systems[system].append(node)
        
        # Compute average strength for each system
        system_strength = {}
        strength_index = self.compute_logical_strength_index()
        
        for system, statements in systems.items():
            if statements:
                avg_strength = sum(strength_index.get(s, 0) for s in statements) / len(statements)
                system_strength[system] = avg_strength
        
        # Compare systems pairwise
        comparisons = {}
        system_names = list(system_strength.keys())
        for i in range(len(system_names)):
            for j in range(i+1, len(system_names)):
                sys1 = system_names[i]
                sys2 = system_names[j]
                if system_strength[sys1] > 0 and system_strength[sys2] > 0:
                    comparisons[(sys1, sys2)] = system_strength[sys1] / system_strength[sys2]
                    
        return comparisons

    # New method added to compute logical strength hierarchy
    def compute_logical_strength_hierarchy(self) -> Dict[str, float]:
        """
        Compute a hierarchy of logical strength for axioms and theorems.
        Higher values indicate stronger logical power.
        """
        strength = {}
        
        # First pass: assign base strength based on number of descendants
        for node in self.G.nodes():
            descendants = set(nx.descendants(self.G, node))
            strength[node] = len(descendants)
        
        # Second pass: adjust for consistency strength relationships
        for source, target, data in self.G.edges(data=True):
            if data.get('relation') == 'Proves' and 'Con' in target:
                # If A proves Con(B), then A is strictly stronger than B
                referenced_system = target.replace('Con(', '').replace(')', '')
                if referenced_system in strength:
                    # Ensure A has higher strength than B
                    strength[source] = max(strength[source], strength.get(referenced_system, 0) + 1)
        
        # Third pass: normalize to [0, 1] range
        max_strength = max(strength.values()) if strength else 1
        for node in strength:
            strength[node] = strength[node] / max_strength
        
        return strength

    # New method added to compute independence likelihood
    def compute_independence_likelihood(self, statement: str, system: str) -> float:
        """
        Estimate the likelihood that a statement is independent from a formal system.
        Returns a value between 0 and 1, where higher values indicate higher likelihood of independence.
        """
        if statement not in self.G.nodes() or system not in self.G.nodes():
            return 0.5  # Default uncertainty when nodes don't exist
        
        # Check if independence is already known
        for _, target, data in self.G.out_edges(system, data=True):
            if target == statement and data.get('relation') == 'Independent':
                return 1.0  # Known to be independent
        
        # Check if provability is already known
        for _, target, data in self.G.out_edges(system, data=True):
            if target == statement and data.get('relation') == 'Proves':
                return 0.0  # Known to be provable
        
        # If not directly connected, use heuristics
        factors = []
        
        # Factor 1: Logical strength comparison
        strength = self.compute_logical_strength_hierarchy()
        system_strength = strength.get(system, 0.5)
        statement_strength = strength.get(statement, 0.5)
        
        # If statement is much stronger than system, independence is more likely
        strength_diff = statement_strength - system_strength
        factors.append(0.5 + (strength_diff * 0.5))
        
        # Factor 2: Existence of similar independence results
        similar_independence = 0
        total_relations = 0
        
        for _, target, data in self.G.out_edges(system, data=True):
            if data.get('relation') == 'Independent':
                similar_independence += 1
            total_relations += 1
        
        independence_ratio = similar_independence / total_relations if total_relations > 0 else 0.5
        factors.append(independence_ratio)
        
        # Combine factors with weights
        weights = [0.6, 0.4]  # Adjust weights as needed
        weighted_sum = sum(f * w for f, w in zip(factors, weights))
        
        return weighted_sum



