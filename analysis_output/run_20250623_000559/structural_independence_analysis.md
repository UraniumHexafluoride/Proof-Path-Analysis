# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| classification   |   count |
|:-----------------|--------:|
| related_unknown  |      25 |
| related_provable |       6 |
| provable         |       2 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| classification   |   degree_centrality |   betweenness_centrality |   closeness_centrality |   pagerank |
|:-----------------|--------------------:|-------------------------:|-----------------------:|-----------:|
| provable         |              0.2031 |                   0.0358 |                 0.1292 |     0.0742 |
| related_provable |              0.0312 |                   0.0000 |                 0.1055 |     0.0298 |
| related_unknown  |              0.0312 |                   0.0022 |                 0.0211 |     0.0269 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| classification   |   pred_systems |   pred_theorems |   succ_systems |   succ_theorems |   independent_neighbors |   provable_neighbors |   neighborhood_size |
|:-----------------|---------------:|----------------:|---------------:|----------------:|------------------------:|---------------------:|--------------------:|
| provable         |           1.50 |            1.00 |           0.00 |            4.00 |                    0.00 |                 2.00 |                6.50 |
| related_provable |           0.00 |            1.00 |           0.00 |            0.00 |                    0.00 |                 1.00 |                1.00 |
| related_unknown  |           0.18 |            0.06 |           0.00 |            0.06 |                    0.00 |                 0.00 |                0.29 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| classification   |   hop |   systems |   theorems |   independent |   provable |   total |
|:-----------------|------:|----------:|-----------:|--------------:|-----------:|--------:|
| provable         |     1 |      0.00 |       4.00 |          0.00 |       1.00 |    4.00 |
| provable         |     2 |      0.00 |       6.00 |          0.00 |       0.00 |    6.00 |
| related_unknown  |     1 |      0.00 |       1.00 |          0.00 |       0.00 |    1.00 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| classification   |   shannon_diversity |
|:-----------------|--------------------:|
| provable         |              1.2432 |
| related_provable |              0.0000 |
| related_unknown  |              0.9591 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated in this run's directory.

