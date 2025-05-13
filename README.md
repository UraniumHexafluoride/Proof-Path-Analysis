# Entailment Cone Analysis

A computational framework for analyzing the logical structure of mathematical knowledge using entailment cones.

## Overview

This project provides tools for analyzing the relationships between mathematical axioms, theorems, and formal systems. It uses graph theory to represent and analyze the structure of mathematical knowledge, with a focus on independence results and logical strength.

## Features

- **Entailment Graph Construction**: Build graphs representing logical relationships between mathematical statements
- **Independence Analysis**: Analyze and predict independence results in mathematical logic
- **Logical Strength Metrics**: Compute metrics for the logical strength of axioms and theorems
- **Open Problem Analysis**: Analyze open problems and suggest potential resolution approaches
- **Visualization**: Generate visualizations of logical dependencies and relationships

## Components

- `entailment_theory.py`: Core classes for logical statements and entailment relations
- `independence_results.py`: Creates entailment graphs with independence results
- `open_problems_analyzer.py`: Tools for analyzing open problems using entailment cones
- `logical_metrics.py`: Metrics for measuring logical strength and relationships
- `structural_analysis.py`: Analysis of structural properties of entailment graphs
- `analyze_open_problems.py`: Script to analyze famous open problems in mathematical logic
- `deep_analysis.py`: In-depth analysis of entailment graph structure

## Usage

To analyze famous open problems:

```python
python analyze_open_problems.py
```

This will generate:
- A comprehensive report in `entailment_output/open_problems_analysis.md`
- Visualizations of problem dependencies in the `entailment_output` directory

## Research Applications

This framework can be used to:
1. Identify critical junctures in the logical structure of mathematics
2. Discover natural divisions in mathematical knowledge
3. Find economical axiom systems for specific theorems
4. Predict independence results
5. Analyze the structure of open problems

## Requirements

- Python 3.7+
- NetworkX
- Matplotlib
- NumPy

## License

[MIT License](LICENSE)