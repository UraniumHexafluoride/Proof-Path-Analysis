# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| Classification | Count |
|---------------|-------|
| related_unknown | 159 |
| provable | 98 |
| related_provable | 12 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| Classification | Degree | Betweenness | Closeness | PageRank |
|---------------|--------|-------------|-----------|----------|
| provable | 0.2699 | 0.0000 | 0.1364 | 0.0073 |
| related_provable | 0.0534 | 0.0001 | 0.0012 | 0.0015 |
| related_unknown | 0.0037 | 0.0000 | 0.0054 | 0.0016 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| Classification | Pred Systems | Pred Theorems | Succ Systems | Succ Theorems | Independent Neighbors | Provable Neighbors |
|---------------|--------------|---------------|--------------|---------------|----------------------|-------------------|
| provable | 0.03 | 37.32 | 0.00 | 37.13 | 0.00 | 73.71 |
| related_provable | 0.08 | 0.08 | 0.00 | 14.58 | 0.00 | 3.67 |
| related_unknown | 0.00 | 1.00 | 0.00 | 0.02 | 0.00 | 0.18 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| Classification | Hop | Systems | Theorems | Independent | Provable | Total |
|---------------|-----|---------|----------|-------------|----------|-------|
| related_unknown | 1 | 0.00 | 1.00 | 0.00 | 0.17 | 1.00 |
| related_unknown | 2 | 0.32 | 13.58 | 0.00 | 3.19 | 13.90 |
| provable | 1 | 0.03 | 37.62 | 0.00 | 36.90 | 37.65 |
| provable | 2 | 0.15 | 7.13 | 0.00 | 1.84 | 7.29 |
| related_provable | 1 | 0.08 | 14.67 | 0.00 | 3.67 | 14.75 |
| related_provable | 2 | 0.25 | 5.17 | 0.00 | 1.00 | 5.42 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| Classification | Average Shannon Diversity |
|---------------|---------------------------|
| related_unknown | 0.7358 |
| provable | 0.2053 |
| related_provable | 0.8492 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated:

1. `centrality_distributions.png`
2. `connectivity_patterns.png`
3. `network_structure.png`
