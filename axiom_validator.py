#!/usr/bin/env python3
"""
Axiom System Validator

This script validates the consistency and correctness of axiom systems and their relationships
across multiple files in the entailment research project.

It performs the following validations:
1. Removes trailing empty lines from CSV files
2. Validates axiom system names and relationships
3. Checks for consistency between different files
4. Creates a dictionary of valid axiom systems and their relationships
"""

import os
import csv
import re
import sys
import networkx as nx
from typing import Dict, List, Set, Tuple, Any

# Constants
VALID_RELATIONS = {"Contains", "Proves", "Independence", "Contradicts", "Disproves", "Unknown"}
BASE_AXIOM_SYSTEMS = {"ZFC", "ZF", "PA", "PA2", "ACA0", "RCA0", "Z", "NBG", "MK", "KP", "Z2", "HOL", "CZF", "IZF", "NF", "NFU", "ML"}
VALID_EXTENSIONS = {"LC", "PD", "MM", "AD", "DC", "Con(PA)", "I0", "I1", "I2"}
VALID_SUFFIXES = {"DC", "AD", "PD", "MM", "LC"}

# List of valid theorems
VALID_THEOREMS = {
    "PH", "GT", "CH", "AC", "Con(PA)", "Con(ZFC)", "GCH", "Ramsey", "KP", "BT", "PD", 
    "MC", "Ind_Comp", "FLT", "4CT", "Con(ZF)", "SH", "PFA", "MM", "V=L", 
    "Fermat's Last Theorem", "Four Color Theorem", "Poincaré Conjecture", "Riemann Hypothesis",
    "P vs NP", "Twin Prime Conjecture", "Goldbach's Conjecture", "Collatz Conjecture",
    "ABC Conjecture", "Hodge Conjecture", "Navier-Stokes Existence and Smoothness",
    "Birch and Swinnerton-Dyer Conjecture", "Gödel's Incompleteness", "Continuum Hypothesis",
    "Axiom of Choice", "Tarski's Undefinability", "Church-Turing Thesis",
    "Halting Problem", "Banach-Tarski Paradox", "Goodstein's Theorem", "Paris-Harrington Theorem"
}

class AxiomValidator:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, "entailment_output")
        self.axiom_systems = {}
        self.theorems = {}
        self.errors = []
        self.warnings = []
        
        # Graph for checking circular dependencies
        self.dependency_graph = nx.DiGraph()
        
    def clean_csv_files(self) -> None:
        """Remove trailing empty lines from all CSV files in the output directory."""
        print("Cleaning CSV files...")
        
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        for filename in csv_files:
            filepath = os.path.join(self.output_dir, filename)
            self._clean_csv_file(filepath)
            
        print(f"Cleaned {len(csv_files)} CSV files")
    
    def _clean_csv_file(self, filepath: str) -> None:
        """Remove trailing empty lines from a CSV file."""
        # Read existing content
        with open(filepath, 'r', newline='') as file:
            lines = file.readlines()
        
        # Remove trailing empty lines
        while lines and lines[-1].strip() == '':
            lines.pop()
        
        # Write back cleaned content
        with open(filepath, 'w', newline='') as file:
            file.writelines(lines)
    
    def validate_axiom_system_names(self) -> None:
        """Validate axiom system names against established patterns."""
        print("Validating axiom system names...")
        
        # Validate from entailment_graph.csv
        graph_file = os.path.join(self.output_dir, "entailment_graph.csv")
        if os.path.exists(graph_file):
            with open(graph_file, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 2:
                        source, target = row[0], row[1]
                        
                        # Check if source and target are valid axiom systems or theorems
                        if self._is_axiom_system(source):
                            self._validate_axiom_system_name(source)
                        
                        if self._is_axiom_system(target):
                            self._validate_axiom_system_name(target)
        
        # Validate from minimal_axioms.csv
        minimal_axioms_file = os.path.join(self.output_dir, "minimal_axioms.csv")
        if os.path.exists(minimal_axioms_file):
            with open(minimal_axioms_file, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 2:
                        theorem = row[0]
                        if theorem and theorem != "Theorem":
                            # Add to theorems dictionary
                            self.theorems[theorem] = True
                            
                        # Check axiom systems in second column
                        if len(row) > 1 and row[1]:
                            axiom_systems = [s.strip() for s in row[1].split(',')]
                            for system in axiom_systems:
                                if system:
                                    self._validate_axiom_system_name(system)
        
        print(f"Validated axiom system names. Found {len(self.errors)} errors.")
        
    def _validate_axiom_system_name(self, name: str) -> bool:
        """
        Validate a single axiom system name.
        Returns True if valid, False otherwise.
        """
        # Check if already validated
        if name in self.axiom_systems:
            return True
            
        # Check if it's a base system
        if name in BASE_AXIOM_SYSTEMS:
            self.axiom_systems[name] = {"type": "base", "extensions": []}
            return True
            
        # Check if it's an extended system
        if "+" in name:
            parts = name.split("+")
            base = parts[0]
            extensions = parts[1:]
            
            # Validate base
            if base not in BASE_AXIOM_SYSTEMS:
                self.errors.append(f"Invalid base axiom system: {base} in {name}")
                return False
                
            # Validate extensions
            for ext in extensions:
                if ext not in VALID_EXTENSIONS and not any(ext.startswith(prefix) for prefix in VALID_EXTENSIONS):
                    self.warnings.append(f"Unusual extension: {ext} in {name}")
            
            self.axiom_systems[name] = {"type": "extended", "base": base, "extensions": extensions}
            return True
            
        # Check if it's a suffixed system
        for suffix in VALID_SUFFIXES:
            if name.endswith(suffix) and name[:-len(suffix)] in BASE_AXIOM_SYSTEMS:
                base = name[:-len(suffix)]
                self.axiom_systems[name] = {"type": "suffixed", "base": base, "suffix": suffix}
                return True
                
        # If we got here, the name is not valid
        self.errors.append(f"Invalid axiom system name: {name}")
        return False
    
    def _is_axiom_system(self, name: str) -> bool:
        """Check if a name represents an axiom system rather than a theorem."""
        # Common axiom system prefixes
        axiom_prefixes = ["ZFC", "ZF", "PA", "ACA", "RCA", "NBG", "MK", "KP", "Z", "HOL", "CZF", "IZF", "NF", "ML"]
        
        # Check if it starts with a common prefix
        return any(name.startswith(prefix) for prefix in axiom_prefixes)
    
    def validate_relationships(self) -> None:
        """Validate relationships between axiom systems and theorems."""
        print("Validating relationships...")
        
        graph_file = os.path.join(self.output_dir, "entailment_graph.csv")
        if os.path.exists(graph_file):
            with open(graph_file, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        source, target, relation = row[0], row[1], row[2]
                        
                        # Check if relation type is valid
                        if relation not in VALID_RELATIONS:
                            self.errors.append(f"Invalid relationship type: {relation} between {source} and {target}")
                        
                        # Add to dependency graph for cycle detection
                        if relation == "Contains":
                            self.dependency_graph.add_edge(source, target)
                        
                        # Check logical consistency
                        if relation in ["Proves", "Contains"] and source == target:
                            self.errors.append(f"Self-reference: {source} {relation} {target}")
                            
                        # Check for contradictory relationships
                        if self._has_contradictory_relationship(source, target, relation):
                            self.errors.append(f"Contradictory relationship: {source} {relation} {target}")
        
        # Check for cycles in the dependency graph
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                for cycle in cycles:
                    self.errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        except nx.NetworkXNoCycle:
            pass  # No cycles, which is good
            
        print(f"Validated relationships. Found {len(self.errors)} errors.")
    
    def _has_contradictory_relationship(self, source: str, target: str, relation: str) -> bool:
        """Check if a relationship contradicts existing relationships."""
        # This would require checking against all existing relationships
        # For this demo, we'll implement a simplified version
        return False
    
    def check_cross_file_consistency(self) -> None:
        """Check consistency of data across multiple files."""
        print("Checking cross-file consistency...")
        
        # Check entailment_graph.csv vs minimal_axioms.csv
        graph_file = os.path.join(self.output_dir, "entailment_graph.csv")
        minimal_axioms_file = os.path.join(self.output_dir, "minimal_axioms.csv")
        
        if os.path.exists(graph_file) and os.path.exists(minimal_axioms_file):
            # Extract theorems and their axiom systems from minimal_axioms.csv
            theorem_to_axioms = {}
            with open(minimal_axioms_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        theorem = row[0]
                        axiom_systems = [s.strip() for s in row[1].split(',')]
                        theorem_to_axioms[theorem] = axiom_systems
            
            # Check against entailment_graph.csv
            with open(graph_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        source, target, relation = row[0], row[1], row[2]
                        
                        # Check if this is a theorem-axiom relationship
                        if relation == "Proves" and target in theorem_to_axioms:
                            if source not in theorem_to_axioms[target]:
                                self.warnings.append(f"Inconsistency: {source} proves {target} in graph but not listed in minimal axioms")
                                
        print(f"Checked cross-file consistency. Found {len(self.warnings)} warnings.")
    
    def create_validated_dictionary(self) -> Dict[str, Any]:
        """Create a dictionary of validated axiom systems and their relationships."""
        print("Creating validated dictionary...")
        
        result = {
            "axiom_systems": self.axiom_systems,
            "theorems": self.theorems,
            "relations": {},
            "validation": {
                "errors": self.errors,
                "warnings": self.warnings
            }
        }
        
        # Add relationship information
        graph_file = os.path.join(self.output_dir, "entailment_graph.csv")
        if os.path.exists(graph_file):
            with open(graph_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        source, target, relation = row[0], row[1], row[2]
                        
                        if source not in result["relations"]:
                            result["relations"][source] = {}
                            
                        if relation not in result["relations"][source]:
                            result["relations"][source][relation] = []
                            
                        result["relations"][source][relation].append(target)
        
        # Write validated data to a new file
        validated_file = os.path.join(self.output_dir, "validated_axiom_systems.csv")
        with open(validated_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["System", "Type", "Base", "Extensions"])
            
            for system, data in self.axiom_systems.items():
                system_type = data.get("type", "unknown")
                base = data.get("base", "")
                extensions = ",".join(data.get("extensions", []))
                writer.writerow([system, system_type, base, extensions])
        
        print(f"Created validated dictionary and saved to {validated_file}")
        return result
    
    def fix_ind_comp_issue(self) -> None:
        """Fix the empty entry for Ind_Comp in minimal_axioms.csv"""
        print("Fixing Ind_Comp issue...")
        
        minimal_axioms_file = os.path.join(self.output_dir, "minimal_axioms.csv")
        if os.path.exists(minimal_axioms_file):
            # Read the file
            with open(minimal_axioms_file, 'r', newline='') as file:
                rows = list(csv.reader(file))
            
            # Find and fix the Ind_Comp row
            for i, row in enumerate(rows):
                if len(row) >= 1 and row[0] == "Ind_Comp":
                    if len(row) < 2 or not row[1]:
                        # Add appropriate axiom systems
                        rows[i] = ["Ind_Comp", "PA2, ACA0"]
                        print("Fixed empty entry for Ind_Comp")
            
            

