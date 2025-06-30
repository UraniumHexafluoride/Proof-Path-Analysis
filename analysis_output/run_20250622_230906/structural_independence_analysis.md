# Structural Analysis of Independence in Mathematics

This report analyzes the structural position of independent statements in the entailment graph.

## Classification Summary

| classification   |   count |
|:-----------------|--------:|
| related_unknown  |     286 |
| related_provable |      33 |
| provable         |       2 |

## Centrality Metrics by Classification

Average centrality metrics for each classification:

| classification   |   degree_centrality |   betweenness_centrality |   closeness_centrality |   pagerank |
|:-----------------|--------------------:|-------------------------:|-----------------------:|-----------:|
| provable         |              0.0578 |                   0.0012 |                 0.0087 |     0.0033 |
| related_provable |              0.0049 |                   0.0000 |                 0.0078 |     0.0014 |
| related_unknown  |              0.0836 |                   0.0000 |                 0.0431 |     0.0033 |

## Neighborhood Structure Analysis

Average neighborhood metrics for each classification:

| classification   |   pred_systems |   pred_theorems |   succ_systems |   succ_theorems |   independent_neighbors |   provable_neighbors |   neighborhood_size |
|:-----------------|---------------:|----------------:|---------------:|----------------:|------------------------:|---------------------:|--------------------:|
| provable         |           1.50 |            0.00 |           0.00 |           17.00 |                    0.00 |                 0.00 |               18.50 |
| related_provable |           0.00 |            1.06 |           0.00 |            0.52 |                    0.00 |                 1.03 |                1.58 |
| related_unknown  |           0.00 |           13.76 |           0.00 |           13.70 |                    0.00 |                 0.00 |               27.46 |

## Extended Neighborhood Analysis

This section analyzes the multi-hop neighborhood structure around theorems.

### Average Neighborhood Composition by Distance

| classification   |   hop |   systems |   theorems |   independent |   provable |   total |
|:-----------------|------:|----------:|-----------:|--------------:|-----------:|--------:|
| provable         |     1 |      0.00 |      17.00 |          0.00 |       0.00 |   17.00 |
| provable         |     2 |      0.00 |      16.00 |          0.00 |       0.00 |   16.00 |
| related_provable |     1 |      0.00 |      17.00 |          0.00 |       0.00 |   17.00 |
| related_unknown  |     1 |      0.00 |      42.79 |          0.00 |       0.00 |   42.79 |
| related_unknown  |     2 |      0.00 |       5.86 |          0.00 |       0.00 |    5.86 |

### Neighborhood Diversity Analysis

Shannon diversity index measures the diversity of node types in the neighborhood.
Higher values indicate more diverse neighborhoods.

| classification   |   shannon_diversity |
|:-----------------|--------------------:|
| provable         |              0.3975 |
| related_provable |              0.0094 |
| related_unknown  |              0.2838 |

## Key Findings

- The comparison of betweenness suggests a difference in how central independent or provable statements are.
- Degree and PageRank differences might indicate varying levels of influence within the network.
- Neighborhood metrics reveal potential clustering of certain classifications.

## Visualizations

The following figures have been generated in this run's directory.

