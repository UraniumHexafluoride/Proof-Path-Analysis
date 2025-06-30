# API Documentation

## Core Module

### Entailment (`mathlogic.core.entailment`)

#### Classes

##### `LogicalStatement`
Represents a formal logical statement or theorem.

```python
class LogicalStatement:
    def __init__(self, symbol: str, description: str = "", formal_system: str = "", is_axiom: bool = False)
```

##### `EntailmentRelation`
Represents a logical relationship between statements.

```python
class EntailmentRelation:
    def __init__(self, source: LogicalStatement, target: LogicalStatement, relation_type: str)
```

##### `EntailmentCone`
Main graph structure for entailment relationships.

```python
class EntailmentCone:
    def __init__(self)
    def add_statement(self, statement: LogicalStatement)
    def add_relation(self, relation: EntailmentRelation)
    def check_closure_properties(self) -> dict
```

### Formal Systems (`mathlogic.core.formal_systems`)

#### Functions

```python
def create_propositional_logic() -> FormalSystem
def create_first_order_logic() -> FormalSystem
def create_zfc() -> FormalSystem
```

### Theorem Prover (`mathlogic.core.theorem_prover`)

#### Classes

##### `TheoremProver`
Handles theorem proving and verification.

```python
class TheoremProver:
    def __init__(self, formal_system: FormalSystem, checker: ProofChecker)
    def prove(self, target: Formula) -> Optional[List[ProofStep]]
```

## Analysis Module

### Structural Analysis (`mathlogic.analysis.structural`)

#### Functions

```python
def compute_structural_metrics(G: nx.DiGraph) -> dict
def analyze_neighborhood_structure(G: nx.DiGraph, classifications: dict) -> dict
def analyze_logical_strength(G: nx.DiGraph, classifications: dict) -> dict
```

### Independence Analysis (`mathlogic.analysis.independence`)

#### Functions

```python
def analyze_independence_patterns(G: nx.DiGraph, metrics: dict) -> dict
def predict_independence_likelihood(G: nx.DiGraph, statement: LogicalStatement) -> float
```

## Graphs Module

### Graph Creation (`mathlogic.graphs.creator`)

#### Functions

```python
def create_entailment_graph(systems: List[str], theorems: List[str], 
                          proves_edges: List[Tuple], independence_edges: List[Tuple]) -> nx.DiGraph
def create_independence_graph(G: nx.DiGraph) -> nx.Graph
```

## Utils Module

### Metrics (`mathlogic.utils.metrics`)

#### Functions

```python
def compute_centrality_metrics(G: nx.DiGraph) -> dict
def compute_neighborhood_metrics(G: nx.DiGraph) -> dict
def compute_logical_strength_metrics(G: nx.DiGraph) -> dict
```

## Data Handling

### CSV Operations (`mathlogic.utils.io`)

#### Functions

```python
def load_graph_from_csv(filename: str) -> nx.DiGraph
def save_graph_to_csv(G: nx.DiGraph, filename: str)
def export_analysis_results(results: dict, filename: str)
```

## Examples

### Basic Usage

```python
from mathlogic.core.entailment import EntailmentCone, LogicalStatement
from mathlogic.core.formal_systems import create_zfc

# Create an entailment cone
cone = EntailmentCone()

# Add statements
zfc = LogicalStatement("ZFC", "Zermelo-Fraenkel with Choice", is_axiom=True)
ch = LogicalStatement("CH", "Continuum Hypothesis")

# Add to cone
cone.add_statement(zfc)
cone.add_statement(ch)

# Analyze
metrics = compute_structural_metrics(cone.graph)
```

### Advanced Analysis

```python
from mathlogic.analysis.structural import analyze_logical_strength
from mathlogic.analysis.independence import predict_independence_likelihood

# Analyze logical strength
strength = analyze_logical_strength(cone.graph, classifications)

# Predict independence
likelihood = predict_independence_likelihood(cone.graph, new_statement)
``` 