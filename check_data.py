import json

with open('entailment_output/scraped_data/improved_scraping_results_20250610_032308.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total entries: {len(data)}')
print(f'First entry keys: {list(data[0].keys()) if data else "No data"}')

# Check first few entries for relationships
for i, entry in enumerate(data[:3]):
    print(f'\nEntry {i}:')
    if 'relationships' in entry:
        print(f'  Has relationships: {len(entry["relationships"])}')
        for j, rel in enumerate(entry['relationships'][:2]):
            print(f'    Rel {j}: {rel}')
    else:
        print(f'  Keys: {list(entry.keys())}')

# Check what types of relationships exist
rel_types = set()
for entry in data:
    if 'relationships' in entry:
        for rel in entry['relationships']:
            if 'type' in rel:
                rel_types.add(rel['type'])

print(f'\nRelationship types found: {rel_types}')

