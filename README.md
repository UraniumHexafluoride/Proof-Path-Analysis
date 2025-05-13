# Entailment Cone Analysis

A computational framework for analyzing the logical structure of mathematical knowledge using entailment cones.

## Overview

This project provides tools for analyzing the relationships between mathematical axioms, theorems, and formal systems. It uses graph theory to represent and analyze the structure of mathematical knowledge, with a focus on independence results and logical strength.

## Current Status

This is an active research project in development. The project is currently in Phase 1 (Data Collection and Validation) and beginning Phase 2 (Structural Analysis) of our implementation plan. We have:

- Created an entailment graph with formal systems, theorems, and logical relations
- Implemented structural metrics for analyzing independence indicators
- Begun analysis of structural positions of independent statements
- Created initial visualizations of the entailment graph

## Research Hypotheses

We are investigating three main hypotheses:

1. **Structural Indicators of Independence**: Independent statements occupy distinctive structural positions in the entailment graph
2. **Minimal Axiom Systems**: There exist natural groupings of theorems based on minimal axiom requirements
3. **Predictive Power of Entailment Structure**: The structure of the entailment graph can predict independence likelihood

Current findings partially support Hypothesis 1, showing that independent statements have higher PageRank and closeness centrality than provable statements.

## Features

- **Entailment Graph Construction**: Build graphs representing logical relationships between mathematical statements
- **Independence Analysis**: Analyze and predict independence results in mathematical logic
- **Logical Strength Metrics**: Compute metrics for the logical strength of axioms and theorems
- **Open Problem Analysis**: Analyze open problems and suggest potential resolution approaches
- **Visualization**: Generate visualizations of logical dependencies and relationships
- **Structural Analysis**: Identify bottlenecks and critical junctures in mathematical knowledge

## Components

- `entailment_theory.py`: Core classes for logical statements and entailment relations
- `independence_results.py`: Creates entailment graphs with independence results
- `open_problems_analyzer.py`: Tools for analyzing open problems using entailment cones
- `logical_metrics.py`: Metrics for measuring logical strength and relationships
- `structural_analysis.py`: Analysis of structural properties of entailment graphs
- `analyze_open_problems.py`: Script to analyze famous open problems in mathematical logic
- `deep_analysis.py`: In-depth analysis of entailment graph structure
- `minimal_axiom_analysis.py`: Analysis of minimal axiom requirements for theorems
- `run_entailment_research.py`: Main script to run the entire research pipeline

## Usage

To run the complete entailment analysis pipeline:

```python
python run_entailment_research.py
```

For specific analyses:

```python
python analyze_open_problems.py  # Analyze famous open problems
python structural_independence_analysis.py  # Analyze structural indicators of independence
python minimal_axiom_analysis.py  # Analyze minimal axiom systems
```

All output files are generated in the `entailment_output` directory.

## Next Steps

Our immediate focus is on:
1. Expanding the graph to reveal more complex structural patterns
2. Implementing comprehensive minimal axiom analysis
3. Developing a predictive model for independence likelihood
4. Validating predictions against known results

## Research Applications

This framework can be used to:
1. Identify critical junctures in the logical structure of mathematics
2. Discover natural divisions in mathematical knowledge
3. Find economical axiom systems for specific theorems
4. Predict independence results
5. Analyze the structure of open problems

## Research Plan

See [research_plan.md](research_plan.md) for our detailed research plan and [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for specific research questions.

## Requirements

- Python 3.7+
- NetworkX
- Matplotlib
- NumPy
- Pandas
- Scikit-learn

## License

[MIT License](LICENSE)
