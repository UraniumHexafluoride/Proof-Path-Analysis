# Mathematical Logic Entailment & Independence Analysis

A computational framework for analyzing the logical structure of mathematical knowledge using entailment graphs, NLP, and advanced structural metrics.

## Current Status

This is an active research project in development. I am currently in Phase 1 (Data Collection and Validation) and beginning Phase 2 (Structural Analysis) of my implementation plan. I have:

- Created an entailment graph with formal systems, theorems, and logical relations
- Implemented structural metrics for analyzing independence indicators
- Begun analysis of structural positions of independent statements
- Created initial visualizations of the entailment graph

## Research Hypotheses

I am investigating three main hypotheses:

1. **Structural Indicators of Independence**: Independent statements occupy distinctive structural positions in the entailment graph
2. **Minimal Axiom Systems**: There exist natural groupings of theorems based on minimal axiom requirements
3. **Predictive Power of Entailment Structure**: The structure of the entailment graph can predict independence likelihood

My current findings are preliminary and inconclusive due to ongoing issues in the pipeline. Further work is needed to validate or refute these hypotheses.

## Features

- **Entailment Graph Construction**: Build graphs representing logical relationships between mathematical statements
- **Independence Analysis**: Analyze and predict independence results in mathematical logic
- **Logical Strength Metrics**: Compute metrics for the logical strength of axioms and theorems
- **Open Problem Analysis**: Analyze open problems and suggest potential resolution approaches
- **Visualization**: Generate visualizations of logical dependencies and relationships
- **Structural Analysis**: Identify bottlenecks and critical junctures in mathematical knowledge
- **Multi-source Theorem Scraping**: Wikipedia, nLab, and more
- **NLP-based Relationship Extraction**: spaCy-powered semantic analysis
- **Comprehensive Reporting**: Markdown reports, CSVs, and visualizations

## Components

- `mathlogic/core/entailment.py`: Core classes for logical statements and entailment relations
- `archive/independence_results.py`: Creates entailment graphs with independence results
- `archive/open_problems_analyzer.py`: Tools for analyzing open problems using entailment cones
- `archive/logical_metrics.py`: Metrics for measuring logical strength and relationships
- `archive/structural_analysis.py`: Analysis of structural properties of entailment graphs
- `archive/analyze_open_problems.py`: Script to analyze famous open problems in mathematical logic
- `archive/deep_analysis.py`: In-depth analysis of entailment graph structure
- `archive/minimal_axiom_analysis.py`: Analysis of minimal axiom requirements for theorems
- `run_entailment_research.py`: Main script to run the entire research pipeline
- `run_analysis.py`: Main script to run the full analysis pipeline
- `run_improved_scraping.py`: Improved scraping with NLP and Wikipedia fallback
- `run_minimal_axiom_analysis.py`: Minimal axiom system analysis

## Directory Layout

```
mathlogic/                # Main package: logic, analysis, scraping, NLP, utils
  ├── core/               # Core logic and entailment classes
  ├── data/               # Data collection, scraping, and processing
  ├── analysis/           # Structural and logical analysis tools
  ├── prediction/         # Predictive modeling (future)
  ├── visualization/      # Visualization tools
  ├── utils/              # Utility functions
  ├── graphs/             # Graph visualization
  └── tests/              # Unit and integration tests
entailment_output/        # Main output: reports, graphs, CSVs, images
analysis_output/          # Run-specific output folders (reports, images, logs)
scraped_data/             # Raw and processed scraped data
archive/                  # Legacy and experimental scripts
examples/                 # Example scripts
run_analysis.py           # Main script to run the full analysis pipeline
run_improved_scraping.py  # Improved scraping with NLP and Wikipedia fallback
run_minimal_axiom_analysis.py # Minimal axiom system analysis
config.json               # Main configuration file
requirements.txt          # Python dependencies
README.md                 # Project documentation
```

## Usage

To run the complete entailment analysis pipeline:

```bash
python run_analysis.py
```

For specific analyses:

```bash
python archive/analyze_open_problems.py  # Analyze famous open problems
python archive/Structural_Independence_Analysis_Ihopeisifinal.py  # Analyze structural indicators of independence
python run_minimal_axiom_analysis.py  # Analyze minimal axiom systems
```

All output files are generated in the `entailment_output/` and `analysis_output/` directories.

## Next Steps

My immediate focus is on:
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

See [research_plan.md](research_plan.md) for my detailed research plan and [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for specific research questions.

## Requirements

- Python 3.8+
- All dependencies in `requirements.txt` (install with `pip install -r requirements.txt`)
- spaCy English model: `python -m spacy download en_core_web_sm`

## License

[MIT License](LICENSE)

