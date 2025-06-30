import json
import networkx as nx
import pickle
import os

# Simple test to check the edge attributes
G = nx.DiGraph()
G.add_node('test_system', type='system')
G.add_node('test_theorem', type='theorem')
G.add_edge('test_system', 'test_theorem', relation='Proves')

# Check the edge data
print('Testing edge attribute structure:')
for u, v, data in G.edges(data=True):
    print(f'Edge: {u} -> {v}')
    print(f'Data: {data}')
    print(f'Relation: {data.get("relation", "NOT_FOUND")}')

# Test the classification logic
node = 'test_theorem'
is_provable = False
is_independent = False

print(f'\nTesting classification for node: {node}')
for pred, _, data in G.in_edges(node, data=True):
    relation = data.get('relation', '')
    print(f'Checking edge from {pred}: relation = "{relation}"')
    if relation == 'Proves':
        is_provable = True
        print('  -> Marked as provable')
    elif relation == 'Independence':
        is_independent = True
        print('  -> Marked as independent')

print(f'Final classification: provable={is_provable}, independent={is_independent}')

# Now let's check the actual graph from the analysis
print('\n' + '='*50)
print('CHECKING ACTUAL GENERATED GRAPH:')
print('='*50)

try:
    # Load the latest scraping data
    import glob
    scraping_files = glob.glob('entailment_output/scraped_data/improved_scraping_results_*.json')
    if scraping_files:
        latest_file = max(scraping_files, key=os.path.getctime)
        print(f'Loading: {latest_file}')
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build a small test graph
        test_G = nx.DiGraph()
        
        # Add some nodes and edges from the data
        count = 0
        for entry in data:
            if count >= 10:  # Just test first 10
                break
                
            if 'relationships' in entry:
                for rel in entry['relationships']:
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', 'related_to')
                    
                    if source and target:
                        test_G.add_node(source, type='theorem')
                        test_G.add_node(target, type='theorem')
                        test_G.add_edge(source, target, relation=rel_type)
                        count += 1
        
        print(f'Test graph: {test_G.number_of_nodes()} nodes, {test_G.number_of_edges()} edges')
        
        # Check edge relations
        relation_types = {}
        for u, v, data in test_G.edges(data=True):
            rel = data.get('relation', 'unknown')
            relation_types[rel] = relation_types.get(rel, 0) + 1
        
        print('Relation types found:')
        for rel_type, count in relation_types.items():
            print(f'  {rel_type}: {count}')
        
        # Test classification on a few nodes
        theorem_nodes = [n for n in test_G.nodes() if test_G.nodes[n].get('type') == 'theorem'][:5]
        print(f'\nTesting classification on {len(theorem_nodes)} theorem nodes:')
        
        for node in theorem_nodes:
            is_provable = False
            is_independent = False
            
            for pred, _, data in test_G.in_edges(node, data=True):
                relation = data.get('relation', '')
                if relation == 'Proves':
                    is_provable = True
                elif relation == 'Independence':
                    is_independent = True
            
            if is_provable and is_independent:
                classification = 'both'
            elif is_provable:
                classification = 'provable'
            elif is_independent:
                classification = 'independent'
            else:
                classification = 'unknown'
            
            print(f'  {node[:50]}...: {classification}')
            
except Exception as e:
    print(f'Error checking actual graph: {e}')

