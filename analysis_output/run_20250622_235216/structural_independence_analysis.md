# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| classification   |   count |
|:-----------------|--------:|
| related_unknown  |     103 |
| related_provable |       6 |
| provable         |       2 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| classification   |   degree_centrality |   betweenness_centrality |   closeness_centrality |   pagerank |
|:-----------------|--------------------:|-------------------------:|-----------------------:|-----------:|
| provable         |              0.0591 |                   0.0030 |                 0.0376 |     0.0067 |
| related_provable |              0.0091 |                   0.0000 |                 0.0307 |     0.0027 |
| related_unknown  |              0.6398 |                   0.0001 |                 0.3219 |     0.0094 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| classification   |   pred_systems |   pred_theorems |   succ_systems |   succ_theorems |   independent_neighbors |   provable_neighbors |   neighborhood_size |
|:-----------------|---------------:|----------------:|---------------:|----------------:|------------------------:|---------------------:|--------------------:|
| provable         |           1.50 |            1.00 |           0.00 |            4.00 |                    0.00 |                 2.00 |                6.50 |
| related_provable |           0.00 |            1.00 |           0.00 |            0.00 |                    0.00 |                 1.00 |                1.00 |
| related_unknown  |           0.03 |           38.03 |           0.00 |           38.03 |                    0.00 |                 0.00 |               76.09 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| classification   |   hop |   systems |   theorems |   independent |   provable |   total |
|:-----------------|------:|----------:|-----------:|--------------:|-----------:|--------:|
| provable         |     1 |      0.00 |       4.00 |          0.00 |       1.00 |    4.00 |
| provable         |     2 |      0.00 |       6.00 |          0.00 |       0.00 |    6.00 |
| related_unknown  |     1 |      0.00 |      45.73 |          0.00 |       0.00 |   45.73 |
| related_unknown  |     2 |      0.00 |       3.85 |          0.00 |       0.00 |    3.85 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| classification   |   shannon_diversity |
|:-----------------|--------------------:|
| provable         |              1.2432 |
| related_provable |              0.0000 |
| related_unknown  |              0.9990 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated in this run's directory.

