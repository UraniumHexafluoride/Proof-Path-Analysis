# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| classification   |   count |
|:-----------------|--------:|
| related_unknown  |     172 |
| related_provable |      18 |
| provable         |       2 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| classification   |   degree_centrality |   betweenness_centrality |   closeness_centrality |   pagerank |
|:-----------------|--------------------:|-------------------------:|-----------------------:|-----------:|
| provable         |              0.0681 |                   0.0027 |                 0.0216 |     0.0037 |
| related_provable |              0.0076 |                   0.0000 |                 0.0193 |     0.0019 |
| related_unknown  |              0.2263 |                   0.0001 |                 0.1145 |     0.0056 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| classification   |   pred_systems |   pred_theorems |   succ_systems |   succ_theorems |   independent_neighbors |   provable_neighbors |   neighborhood_size |
|:-----------------|---------------:|----------------:|---------------:|----------------:|------------------------:|---------------------:|--------------------:|
| provable         |           1.50 |            1.00 |           0.00 |           10.50 |                    0.00 |                 2.00 |               13.00 |
| related_provable |           0.00 |            1.44 |           0.00 |            0.00 |                    0.00 |                 1.06 |                1.44 |
| related_unknown  |           0.02 |           22.58 |           0.00 |           22.62 |                    0.00 |                 0.00 |               45.22 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| classification   |   hop |   systems |   theorems |   independent |   provable |   total |
|:-----------------|------:|----------:|-----------:|--------------:|-----------:|--------:|
| provable         |     1 |      0.00 |      10.50 |          0.00 |       1.00 |   10.50 |
| provable         |     2 |      0.00 |       8.50 |          0.00 |       0.00 |    8.50 |
| related_unknown  |     1 |      0.00 |      39.47 |          0.00 |       0.00 |   39.47 |
| related_unknown  |     2 |      0.00 |       4.11 |          0.00 |       0.00 |    4.11 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| classification   |   shannon_diversity |
|:-----------------|--------------------:|
| provable         |              0.9127 |
| related_provable |              0.0000 |
| related_unknown  |              0.5003 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated in this run's directory.

