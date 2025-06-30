import networkx as nx
import re

# 1. Transitive Redundancy Removal
def remove_transitive_edges(G: nx.DiGraph, edge_type: str = None, inplace: bool = False):
    """
    Remove edges that are transitively implied by other paths.
    If edge_type is specified, only consider edges with that relation_type.
    Returns a new graph (unless inplace=True).
    """
    if not inplace:
        G = G.copy()
    to_remove = []
    for u, v, data in G.edges(data=True):
        if edge_type and data.get('relation_type') != edge_type:
            continue
        G.remove_edge(u, v)
        if nx.has_path(G, u, v):
            to_remove.append((u, v))
        else:
            G.add_edge(u, v, **data)
    if inplace:
        for u, v in to_remove:
            G.remove_edge(u, v)
        return G
    else:
        for u, v in to_remove:
            G.remove_edge(u, v)
        return G

# 2. Node/Relationship Normalization
def canonicalize_label(label: str) -> str:
    """
    Normalize theorem/statement labels (case, whitespace, common synonyms).
    """
    label = label.strip().lower()
    label = re.sub(r'[_\s]+', ' ', label)
    synonyms = {
        "iff": "if and only if",
        "equivalent to": "if and only if",
        "implies": "implies",
        "follows from": "implies"
    }
    for k, v in synonyms.items():
        label = label.replace(k, v)
    return label

def normalize_graph_nodes(G: nx.DiGraph):
    """
    Canonicalize all node labels in the graph.
    Returns a mapping from old to new labels.
    """
    mapping = {}
    for node in list(G.nodes()):
        new_label = canonicalize_label(node)
        if new_label != node:
            mapping[node] = new_label
    nx.relabel_nodes(G, mapping, copy=False)
    return mapping

# 3. Disconnected Subgraph Diagnostics
def graph_diagnostics(G: nx.DiGraph):
    """
    Print diagnostics about disconnected components, orphans, and cycles.
    """
    print("=== Graph Diagnostics ===")
    weak_components = list(nx.weakly_connected_components(G))
    print(f"Number of weakly connected components: {len(weak_components)}")
    for i, comp in enumerate(weak_components):
        print(f"  Component {i+1}: {len(comp)} nodes")
    orphans = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    print(f"Orphaned nodes: {orphans}")
    systems = [n for n, d in G.nodes(data=True) if d.get('type') == 'formal_system']
    theorems = [n for n, d in G.nodes(data=True) if d.get('type') == 'theorem']
    for sys in systems:
        reachable = set(nx.descendants(G, sys))
        unreachable = [t for t in theorems if t not in reachable]
        if len(unreachable) == len(theorems):
            print(f"Formal system '{sys}' cannot reach any theorems!")
    cycles = list(nx.simple_cycles(G))
    print(f"Number of cycles: {len(cycles)}")
    if cycles:
        print("Sample cycles:", cycles[:3])

# 4. Edge Confidence Normalization and Source Tagging
def normalize_edge_confidence(G: nx.DiGraph, min_conf=0.0, max_conf=1.0):
    """
    Normalize all edge confidence values to [min_conf, max_conf].
    """
    confidences = [d.get('confidence', 0.5) for _, _, d in G.edges(data=True)]
    if not confidences:
        return
    min_c, max_c = min(confidences), max(confidences)
    for u, v, d in G.edges(data=True):
        conf = d.get('confidence', 0.5)
        if max_c > min_c:
            norm_conf = min_conf + (conf - min_c) * (max_conf - min_conf) / (max_c - min_c)
        else:
            norm_conf = conf
        d['confidence'] = norm_conf

def tag_edge_source(G: nx.DiGraph, default_source="manual"):
    """
    Ensure every edge has a 'source_origin' tag.
    """
    for u, v, d in G.edges(data=True):
        if 'source_origin' not in d:
            d['source_origin'] = default_source 