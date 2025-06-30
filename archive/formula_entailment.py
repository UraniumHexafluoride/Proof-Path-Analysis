import re
import itertools
import multiprocessing
from enum import Enum
from typing import List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# -----------------------------
# Data Structures and Classes
# -----------------------------

class FormulaType(Enum):
    ATOM = 1
    NEGATION = 2
    CONJUNCTION = 3
    DISJUNCTION = 4
    IMPLICATION = 5
    EQUIVALENCE = 6

class Formula:
    def __init__(self, formula_type: FormulaType, content: str = None, 
                 left: 'Formula' = None, right: 'Formula' = None):
        self.type = formula_type
        self.content = content    # For atoms
        self.left = left          # For binary operators (or inner for negation)
        self.right = right        # For binary operators
        self.inner = left         # Alias for unary operators
        
    def __str__(self) -> str:
        if self.type == FormulaType.ATOM:
            return self.content
        elif self.type == FormulaType.NEGATION:
            return f"¬({self.inner})"
        elif self.type == FormulaType.CONJUNCTION:
            return f"({self.left} ∧ {self.right})"
        elif self.type == FormulaType.DISJUNCTION:
            return f"({self.left} ∨ {self.right})"
        elif self.type == FormulaType.IMPLICATION:
            return f"({self.left} → {self.right})"
        elif self.type == FormulaType.EQUIVALENCE:
            return f"({self.left} ↔ {self.right})"
        return "Invalid Formula"
    
    def __eq__(self, other):
        if not isinstance(other, Formula):
            return False
        if self.type != other.type:
            return False
        if self.type == FormulaType.ATOM:
            return self.content == other.content
        elif self.type == FormulaType.NEGATION:
            return self.inner == other.inner
        else:
            return self.left == other.left and self.right == other.right
    
    def __hash__(self):
        if self.type == FormulaType.ATOM:
            return hash((self.type, self.content))
        elif self.type == FormulaType.NEGATION:
            return hash((self.type, hash(self.inner)))
        else:
            return hash((self.type, hash(self.left), hash(self.right)))

class FormulaParser:
    def parse(self, formula_str: str) -> Formula:
        formula_str = formula_str.strip()
        # Atomic formula
        if len(formula_str) == 1 and formula_str.isalpha():
            return Formula(FormulaType.ATOM, content=formula_str)
        # Negation
        if formula_str.startswith("¬"):
            inner = self.parse(formula_str[1:])
            return Formula(FormulaType.NEGATION, left=inner)
        # Remove outer parentheses if present
        if formula_str.startswith("(") and formula_str.endswith(")"):
            formula_str = formula_str[1:-1].strip()
            paren_count = 0
            for i, char in enumerate(formula_str):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                if paren_count == 0:
                    if char == "∧":
                        left = self.parse(formula_str[:i].strip())
                        right = self.parse(formula_str[i+1:].strip())
                        return Formula(FormulaType.CONJUNCTION, left=left, right=right)
                    if char == "∨":
                        left = self.parse(formula_str[:i].strip())
                        right = self.parse(formula_str[i+1:].strip())
                        return Formula(FormulaType.DISJUNCTION, left=left, right=right)
                    if char == "→" or (char == "-" and i+1 < len(formula_str) and formula_str[i+1] == ">"):
                        left = self.parse(formula_str[:i].strip())
                        right = self.parse(formula_str[i+1:].strip() if char == "→" else formula_str[i+2:].strip())
                        return Formula(FormulaType.IMPLICATION, left=left, right=right)
                    if char == "↔" or (char == "<" and i+2 < len(formula_str) and formula_str[i+1:i+3] == "->"):
                        left = self.parse(formula_str[:i].strip())
                        right = self.parse(formula_str[i+1:].strip() if char == "↔" else formula_str[i+3:].strip())
                        return Formula(FormulaType.EQUIVALENCE, left=left, right=right)
        raise ValueError(f"Malformed formula: {formula_str}")

class InferenceRule:
    def __init__(self, name: str, apply_func):
        self.name = name
        self.apply_func = apply_func

class EntailmentEngine:
    def __init__(self, parser: FormulaParser = None):
        self.parser = parser or FormulaParser()
        self.formulas: Set[Formula] = set()
        self.entailment_graph = nx.DiGraph()
        self.rules: List[InferenceRule] = []
        self._init_rules()
    
    def _init_rules(self):
        self.rules = [
            InferenceRule("Modus Ponens", self.apply_modus_ponens),
            InferenceRule("Modus Tollens", self.apply_modus_tollens),
            InferenceRule("Conjunction Introduction", self.apply_conjunction_introduction),
            InferenceRule("Conjunction Elimination", self.apply_conjunction_elimination),
            InferenceRule("Disjunction Introduction", self.apply_disjunction_introduction),
            InferenceRule("Disjunction Elimination", self.apply_disjunction_elimination),
            InferenceRule("Double Negation", self.apply_double_negation),
            InferenceRule("Contraposition", self.apply_contraposition)
        ]
    
    def add_formula(self, formula_str: str) -> Formula:
        formula = self.parser.parse(formula_str)
        self.formulas.add(formula)
        self.entailment_graph.add_node(str(formula))
        return formula

    def apply_rule_parallel(self, rule: InferenceRule, formulas: Set[Formula]) -> Set[Formula]:
        """Apply a rule in parallel across multiple processes."""
        cpu_count = multiprocessing.cpu_count()
        formula_list = list(formulas)
        chunk_size = max(1, len(formula_list) // cpu_count)
        chunks = [formula_list[i:i + chunk_size] for i in range(0, len(formula_list), chunk_size)]
        
        with ProcessPoolExecutor() as executor:
            process_chunk = partial(self._process_chunk, rule_func=rule.apply_func)
            results = executor.map(process_chunk, chunks)
        
        new_formulas = set()
        for result in results:
            new_formulas.update(result)
        return new_formulas

    def _process_chunk(self, chunk: List[Formula], rule_func) -> Set[Formula]:
        """Process a chunk of formulas with the given rule function."""
        chunk_set = set(chunk)
        return rule_func(chunk_set)
    
    def apply_all_rules_parallel(self, max_iterations: int = 5) -> Set[Formula]:
        """Apply all inference rules in parallel for better performance."""
        print("Applying inference rules in parallel...")
        new_formulas = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration_new = set()
            
            for rule in self.rules:
                derived = self.apply_rule_parallel(rule, self.formulas)
                
                for formula in derived:
                    if formula not in self.formulas:
                        iteration_new.add(formula)
                        formula_str = str(formula)
                        self.entailment_graph.add_node(formula_str)
                        
                        # Find and connect premises
                        premises = self._find_premises(formula, rule.name)
                        for premise in premises:
                            self.entailment_graph.add_edge(str(premise), formula_str, rule=rule.name)
            
            if not iteration_new:
                break
                
            self.formulas.update(iteration_new)
            new_formulas.update(iteration_new)
            iteration += 1
            print(f"Iteration {iteration}: {len(iteration_new)} new formulas derived.")
        
        return new_formulas

    def find_premises(self, conclusion: Formula, rule_name: str) -> List[Formula]:
        premises = []
        if rule_name == "Modus Ponens":
            for formula in self.formulas:
                if formula.type == FormulaType.IMPLICATION and formula.right == conclusion and formula.left in self.formulas:
                    premises.append(formula)
                    premises.append(formula.left)
        elif rule_name == "Conjunction Elimination":
            for formula in self.formulas:
                if formula.type == FormulaType.CONJUNCTION and (formula.left == conclusion or formula.right == conclusion):
                    premises.append(formula)
        return premises

    # -----------------------------
    # Rule Application Methods
    # -----------------------------
    def apply_modus_ponens(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.IMPLICATION:
                for other in formulas:
                    if f.left == other:
                        new_formulas.add(f.right)
        return new_formulas

    def apply_modus_tollens(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.IMPLICATION:
                for other in formulas:
                    if other.type == FormulaType.NEGATION and other.inner == f.right:
                        new_formulas.add(Formula(FormulaType.NEGATION, left=f.left))
        return new_formulas

    def apply_conjunction_introduction(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        formula_list = list(formulas)
        max_conj = 100
        formula_list.sort(key=lambda x: 1 if x.type == FormulaType.ATOM else 2)
        count = 0
        for i in range(len(formula_list)):
            if count >= max_conj:
                break
            for j in range(i + 1, len(formula_list)):
                # Avoid nesting conjunctions unnecessarily.
                if formula_list[i].type == FormulaType.CONJUNCTION and formula_list[j].type == FormulaType.CONJUNCTION:
                    continue
                new_formulas.add(Formula(FormulaType.CONJUNCTION, left=formula_list[i], right=formula_list[j]))
                count += 1
                if count >= max_conj:
                    break
        return new_formulas

    def apply_conjunction_elimination(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.CONJUNCTION:
                new_formulas.add(f.left)
                new_formulas.add(f.right)
        return new_formulas

    def apply_disjunction_introduction(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        formula_list = list(formulas)
        max_disj = 100
        formula_list.sort(key=lambda x: 1 if x.type == FormulaType.ATOM else 2)
        count = 0
        for i in range(min(10, len(formula_list))):
            if count >= max_disj:
                break
            for j in range(min(10, len(formula_list))):
                if i != j:
                    new_formulas.add(Formula(FormulaType.DISJUNCTION, left=formula_list[i], right=formula_list[j]))
                    count += 1
                    if count >= max_disj:
                        break
        return new_formulas

    def apply_disjunction_elimination(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.DISJUNCTION:
                for other in formulas:
                    if other.type == FormulaType.NEGATION and other.inner == f.left:
                        new_formulas.add(f.right)
                    elif other.type == FormulaType.NEGATION and other.inner == f.right:
                        new_formulas.add(f.left)
        return new_formulas

    def apply_double_negation(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.NEGATION and f.inner and f.inner.type == FormulaType.NEGATION:
                new_formulas.add(f.inner.inner)
        return new_formulas

    def apply_contraposition(self, formulas: Set[Formula]) -> Set[Formula]:
        new_formulas = set()
        for f in formulas:
            if f.type == FormulaType.IMPLICATION:
                not_b = Formula(FormulaType.NEGATION, left=f.right)
                not_a = Formula(FormulaType.NEGATION, left=f.left)
                new_formulas.add(Formula(FormulaType.IMPLICATION, left=not_b, right=not_a))
        return new_formulas

    # -----------------------------
    # Visualization Methods
    # -----------------------------
    def visualize_entailment_graph(self, output_path: str = None):
        plt.figure(figsize=(16, 12))
        try:
            pos = nx.nx_agraph.graphviz_layout(self.entailment_graph, prog='dot')
        except:
            # Fallback hierarchical layout
            pos = nx.spring_layout(self.entailment_graph, seed=42)
        # Color nodes based on whether they are atoms or complex formulas
        atom_nodes = [n for n in self.entailment_graph.nodes() if "(" not in n]
        complex_nodes = [n for n in self.entailment_graph.nodes() if "(" in n]
        nx.draw_networkx_nodes(self.entailment_graph, pos, nodelist=atom_nodes,
                               node_color='lightgreen', node_size=800, alpha=0.8)
        nx.draw_networkx_nodes(self.entailment_graph, pos, nodelist=complex_nodes,
                               node_color='lightblue', node_size=1000, alpha=0.8)
        # Draw edges grouped by rule (using tab10 colormap)
        rule_set = {data.get('rule', 'Unknown') for _,_,data in self.entailment_graph.edges(data=True)}
        for rule in rule_set:
            edges = [(u, v) for u, v, data in self.entailment_graph.edges(data=True)
                     if data.get('rule', 'Unknown') == rule]
            if edges:
                nx.draw_networkx_edges(self.entailment_graph, pos, edgelist=edges,
                                       width=2, alpha=0.7,
                                       edge_color=plt.cm.tab10(hash(rule) % 10))
        nx.draw_networkx_labels(self.entailment_graph, pos, font_size=10, font_weight='bold')
        plt.title("Entailment Graph", fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_formula_type_regions(self, output_path: str = None):
        # This method creates overlapping regions for each formula type.
        import matplotlib.patches as mpatches
        import numpy as np
        plt.figure(figsize=(16, 12), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        formulas = list(self.entailment_graph.nodes())
        formula_groups = {}
        for f in formulas:
            f_type = get_formula_type(f)
            formula_groups.setdefault(f_type, []).append(f)
        region_positions = {
            "atom": (0.8, 0.8),
            "implication": (0.2, 0.2),
            "conjunction": (0.8, 0.2),
            "disjunction": (0.2, 0.8),
            "negation": (0.3, 0.5),
            "equivalence": (0.5, 0.5),
            "other": (0.7, 0.5)
        }
        region_sizes = {}
        for typ, group in formula_groups.items():
            region_sizes[typ] = max(0.15, 0.05 + 0.02 * len(group))
        for typ, pos in region_positions.items():
            if typ in formula_groups and formula_groups[typ]:
                size = region_sizes.get(typ, 0.15)
                circle = plt.Circle(pos, size, color='lightblue', alpha=0.5)
                ax.add_patch(circle)
                plt.text(pos[0], pos[1], typ.capitalize(), ha='center', va='center',
                         fontsize=14, fontweight='bold')
        # Draw rule connections between regions
        for u, v, data in self.entailment_graph.edges(data=True):
            rule = data.get('rule', 'Unknown')
            u_type = get_formula_type(u)
            v_type = get_formula_type(v)
            if u_type in region_positions and v_type in region_positions:
                u_pos = region_positions[u_type]
                v_pos = region_positions[v_type]
                plt.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], color='gray', alpha=0.3,
                         linestyle='--', linewidth=1)
                mid_x, mid_y = (u_pos[0] + v_pos[0]) / 2, (u_pos[1] + v_pos[1]) / 2
                plt.text(mid_x, mid_y, rule, ha='center', va='center', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.title("Overview of Entailment Graph Structure", fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Stub methods for additional visualizations
    def visualize_proof_space_topology(self, output_path: str = None):
        print("Visualizing proof space topology... (Not Implemented)")
        # Implement your topology visualization here.
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

    def visualize_axiom_system_graph(self, output_path: str = None):
        print("Visualizing axiom system graph... (Not Implemented)")
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

    def visualize_metamodeling_graph(self, output_path: str = None):
        print("Visualizing metamodeling graph... (Not Implemented)")
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

    def visualize_wolfram_style_graph(self, output_path: str = None):
        print("Visualizing Wolfram-style graph... (Not Implemented)")
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

# -----------------------------
# Helper Function
# -----------------------------
def get_formula_type(formula_str: str) -> str:
    if "→" in formula_str or "->" in formula_str:
        return "implication"
    elif "∧" in formula_str:
        return "conjunction"
    elif "∨" in formula_str:
        return "disjunction"
    elif "¬" in formula_str:
        return "negation"
    elif "↔" in formula_str or "<->" in formula_str:
        return "equivalence"
    elif len(formula_str) <= 2 and formula_str.isalpha():
        return "atom"
    else:
        return "other"

# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    # Initialize engine and add some starting formulas.
    engine = EntailmentEngine()
    starting_formulas = ["P", "Q", "(P → R)", "(Q → S)", "((R ∧ S) → T)"]
    for f in starting_formulas:
        engine.add_formula(f)
    # Optionally add more derived formulas (stub)
    interesting_formulas = []  # Replace with your own list if available.
    for f in interesting_formulas:
        engine.add_formula(str(f))

    print("Applying inference rules...")
    engine.apply_all_rules(max_iterations=5)
    
    print("Visualizing full entailment graph...")
    engine.visualize_entailment_graph(output_path="entailment_graph.png")
    
    print("Visualizing formula type regions...")
    engine.visualize_formula_type_regions(output_path="formula_type_regions.png")
    
    print("Visualizing proof space topology...")
    engine.visualize_proof_space_topology(output_path="proof_space_topology.png")
    
    print("Visualizing axiom system graph...")
    engine.visualize_axiom_system_graph(output_path="axiom_system_graph.png")
    
    print("Visualizing metamodeling graph...")
    engine.visualize_metamodeling_graph(output_path="metamodeling_graph.png")
    
    print("Visualizing Wolfram-style graph...")
    engine.visualize_wolfram_style_graph(output_path="wolfram_style_graph.png")
    
    # Create a subgraph for key inference paths
    key_formulas = ["P", "Q", "R", "S", "T", "(P → R)", "(Q → S)", "((R ∧ S) → T)"]
    subgraph = nx.DiGraph()
    for f in key_formulas:
        if str(engine.parser.parse(f)) in engine.entailment_graph.nodes():
            subgraph.add_node(str(engine.parser.parse(f)))
    for source in subgraph.nodes():
        for target in subgraph.nodes():
            if engine.entailment_graph.has_edge(source, target):
                data = engine.entailment_graph.get_edge_data(source, target)
                subgraph.add_edge(source, target, **data)
    plt.figure(figsize=(16, 12))
    try:
        pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
    except:
        # Fall back to a custom layered layout
        layers = {}
        roots = [node for node in subgraph.nodes() if subgraph.in_degree(node)==0]
        for node in subgraph.nodes():
            min_dist = float('inf')
            for root in roots:
                try:
                    d = nx.shortest_path_length(subgraph, root, node)
                    min_dist = min(min_dist, d)
                except nx.NetworkXNoPath:
                    pass
            layers[node] = min_dist if min_dist < float('inf') else 0
        pos = {}
        layer_groups = itertools.groupby(sorted(layers.items(), key=lambda x: x[1]), key=lambda x: x[1])
        for layer, nodes in layer_groups:
            nodes = list(nodes)
            n_nodes = len(nodes)
            for i, (node, _) in enumerate(nodes):
                pos[node] = (2*(i - n_nodes/2), -layer*3)
    atom_nodes = [n for n in subgraph.nodes() if len(n)==1]
    complex_nodes = [n for n in subgraph.nodes() if len(n)>1]
    nx.draw_networkx_nodes(subgraph, pos, nodelist=atom_nodes, node_color='lightgreen', node_size=800, alpha=0.8)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=complex_nodes, node_color='lightblue', node_size=1200, alpha=0.8)
    # Create legend for node types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Atomic Formulas'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Complex Formulas')
    ]
    # Draw edges by rule colors
    rule_colors = {
        'Modus Ponens': 'red',
        'Conjunction Introduction': 'blue',
        'Conjunction Elimination': 'green',
        'Disjunction Introduction': 'purple',
        'Disjunction Elimination': 'orange',
        'Double Negation': 'brown',
        'Contraposition': 'pink'
    }
    for rule, color in rule_colors.items():
        rule_edges = [(u, v) for u, v, data in subgraph.edges(data=True) 
                      if data.get('rule', 'Unknown') == rule]
        if rule_edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=rule_edges,
                                   width=2, alpha=0.7, edge_color=color)
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=rule))
    # Draw remaining edges
    other_edges = [(u, v) for u, v, data in subgraph.edges(data=True)
                   if data.get('rule', 'Unknown') not in rule_colors]
    if other_edges:
        nx.draw_networkx_edges(subgraph, pos, edgelist=other_edges,
                               width=1.5, alpha=0.5, edge_color='gray')
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, label='Other Rules'))
    nx.draw_networkx_labels(subgraph, pos, font_size=12, font_weight='bold')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title("Key Inference Paths in Entailment Graph", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("entailment_key_paths.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()





















