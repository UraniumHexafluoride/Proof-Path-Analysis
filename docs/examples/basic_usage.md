# Basic Usage Guide

This guide demonstrates common usage patterns for the Mathematical Logic Analysis Framework.

## Getting Started

### Installation

```bash
# Install from requirements
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Example

```python
from mathlogic.core.entailment import EntailmentCone, LogicalStatement
from mathlogic.analysis.structural import StructuralAnalyzer

# Create basic statements
zfc = LogicalStatement("ZFC", "Zermelo-Fraenkel set theory with Choice", "Set Theory", True)
ch = LogicalStatement("CH", "Continuum Hypothesis", "Set Theory", False)
ac = LogicalStatement("AC", "Axiom of Choice", "Set Theory", True)

# Create the entailment cone
cone = EntailmentCone()

# Add statements
cone.add_statement(zfc)
cone.add_statement(ch)
cone.add_statement(ac)

# Add relationships
from mathlogic.core.entailment import EntailmentRelation
cone.add_relation(EntailmentRelation(zfc, ac, "Contains"))
cone.add_relation(EntailmentRelation(zfc, ch, "Independence"))

# Analyze the structure
analyzer = StructuralAnalyzer(cone)
results = analyzer.analyze_structure()

# Print results
print("Classifications:")
for statement, classification in results.classifications.items():
    print(f"{statement}: {classification}")

print("\nRecommendations:")
for rec in results.recommendations:
    print(f"- {rec}")
```

### Basic Visualization

```python
from mathlogic.graphs.visualization import visualize_graph

# Create basic visualization
visualize_graph(results.graph, "entailment_graph.png")
```

## Common Tasks

### Adding Multiple Statements

```python
# Add multiple statements at once
statements = [
    LogicalStatement("PA", "Peano Arithmetic", "Number Theory", True),
    LogicalStatement("Con(PA)", "Consistency of PA", "Meta-mathematics", False),
    LogicalStatement("GCH", "Generalized Continuum Hypothesis", "Set Theory", False)
]

for stmt in statements:
    cone.add_statement(stmt)
```

### Checking Relations

```python
# Check if statements are related
has_relation = cone.check_relation(zfc, ch)
if has_relation:
    print(f"Relation type: {has_relation.relation_type}")
```

### Computing Basic Metrics

```python
# Get basic metrics for a statement
metrics = results.metrics['centrality'][zfc.symbol]
print(f"Centrality: {metrics['degree']}")
print(f"Betweenness: {metrics['betweenness']}")
```

## Error Handling

```python
try:
    # Try to add invalid relation
    invalid_relation = EntailmentRelation(ch, ch, "Invalid")
    cone.add_relation(invalid_relation)
except ValueError as e:
    print(f"Error: {e}")
```

## Basic Analysis

### Independence Analysis

```python
# Check for independence patterns
independent_statements = [
    node for node, cls in results.classifications.items()
    if cls == 'independent'
]
print("Independent statements:", independent_statements)
```

### Structural Properties

```python
# Check closure properties
properties = cone.check_closure_properties()
print("Transitivity:", properties['transitivity'])
print("Reflexivity:", properties['reflexivity'])
print("Consistency:", properties['consistency'])
```

## Next Steps

- Check the [Advanced Usage Guide](advanced_usage.md) for more complex scenarios
- Review the [API Reference](../api/core.md) for detailed information
- Explore [Examples](../examples/) for more use cases

