# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| classification   |   count |
|:-----------------|--------:|
| related_unknown  |     198 |
| related_provable |       6 |
| provable         |       2 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| classification   |   degree_centrality |   betweenness_centrality |   closeness_centrality |   pagerank |
|:-----------------|--------------------:|-------------------------:|-----------------------:|-----------:|
| provable         |              0.0366 |                   0.0008 |                 0.0305 |     0.0153 |
| related_provable |              0.0138 |                   0.0001 |                 0.0208 |     0.0083 |
| related_unknown  |              0.0022 |                   0.0000 |                 0.0014 |     0.0046 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| classification   |   pred_systems |   pred_theorems |   succ_systems |   succ_theorems |   independent_neighbors |   provable_neighbors |   neighborhood_size |
|:-----------------|---------------:|----------------:|---------------:|----------------:|------------------------:|---------------------:|--------------------:|
| provable         |           1.50 |            2.50 |           0.00 |            3.50 |                    0.00 |                 1.00 |                7.50 |
| related_provable |           0.00 |            1.67 |           0.00 |            1.17 |                    0.00 |                 1.50 |                2.83 |
| related_unknown  |           0.02 |            0.17 |           0.00 |            0.17 |                    0.00 |                 0.01 |                0.36 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| classification   |   hop |   systems |   theorems |   independent |   provable |   total |
|:-----------------|------:|----------:|-----------:|--------------:|-----------:|--------:|
| provable         |     1 |      0.00 |       7.00 |          0.00 |       1.00 |    7.00 |
| provable         |     2 |      0.00 |       2.00 |          0.00 |       0.00 |    2.00 |
| related_provable |     1 |      0.00 |       2.33 |          0.00 |       1.00 |    2.33 |
| related_provable |     2 |      0.00 |       5.33 |          0.00 |       1.00 |    5.33 |
| related_unknown  |     1 |      0.00 |       1.32 |          0.00 |       0.04 |    1.32 |
| related_unknown  |     2 |      0.00 |       1.67 |          0.00 |       0.22 |    1.67 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| classification   |   shannon_diversity |
|:-----------------|--------------------:|
| provable         |              1.0995 |
| related_provable |              0.4637 |
| related_unknown  |              0.2198 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated in this run's directory.

