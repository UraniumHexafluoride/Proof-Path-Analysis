"""
Data integration module for combining scraped mathematical data with existing dataset.
"""

import json
import os
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict

def load_scraped_data(filepath: str) -> List[Dict[str, Any]]:
    """Load scraped data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('scraped_pages', [])
    except Exception as e:
        print(f"Error loading scraped data: {e}")
        return []

def normalize_relationship(rel_type: str) -> str:
    """Normalize relationship types to match our schema."""
    # Map scraped relationship types to our schema
    relationship_mapping = {
        'proves': 'Proves',
        'implies': 'Implies',
        'equivalent': 'Equivalent',
        'independent': 'Independence',
        'related': 'Related',
        'uses': 'Uses',
        'mentions': 'Related'  # Default mapping for mentions
    }
    return relationship_mapping.get(rel_type.lower(), 'Related')

def validate_theorem_data(theorem_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate theorem data for integration."""
    required_fields = ['title', 'type', 'relationships', 'metadata']
    
    # Check required fields
    for field in required_fields:
        if field not in theorem_data:
            return False, f"Missing required field: {field}"
    
    # Validate relationships
    if not isinstance(theorem_data['relationships'], list):
        return False, "Relationships must be a list"
    
    # Validate metadata
    metadata = theorem_data.get('metadata', {})
    if not isinstance(metadata, dict):
        return False, "Metadata must be a dictionary"
    
    required_metadata = ['description', 'field', 'complexity']
    for field in required_metadata:
        if field not in metadata:
            return False, f"Missing required metadata field: {field}"
    
    return True, "Valid"

def merge_theorem_data(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new theorem data with existing data, preserving important information."""
    merged = existing.copy()
    
    # Initialize metadata if it doesn't exist
    if 'metadata' not in merged:
        merged['metadata'] = {}
    if 'metadata' not in new:
        new['metadata'] = {}
    
    # Update description if new one is more detailed
    if len(new.get('metadata', {}).get('description', '')) > len(existing.get('metadata', {}).get('description', '')):
        merged['metadata']['description'] = new['metadata']['description']
    
    # Update field if new one is more specific
    if new.get('metadata', {}).get('field', 'Unknown') != 'Unknown':
        merged['metadata']['field'] = new['metadata']['field']
    
    # Update complexity if new one is available
    if new.get('metadata', {}).get('complexity', 'unknown') != 'unknown':
        merged['metadata']['complexity'] = new['metadata']['complexity']
    
    # Merge relationships, avoiding duplicates
    existing_rels = {(r['source'], r['target'], r['relation_type']) 
                    for r in existing.get('relationships', [])}
    
    for rel in new.get('relationships', []):
        rel_key = (rel['source'], rel['target'], rel['relation_type'])
        if rel_key not in existing_rels:
            merged.setdefault('relationships', []).append(rel)
    
    return merged

def integrate_scraped_data(scraped_data: List[Dict[str, Any]], 
                         existing_theorems: Dict[str, Any],
                         existing_systems: Dict[str, Any],
                         existing_relationships: List[Tuple[str, str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[str, str, str]]]:
    """
    Integrate scraped data with existing dataset.
    
    Args:
        scraped_data: List of scraped theorem/system data
        existing_theorems: Current theorem database
        existing_systems: Current system database
        existing_relationships: Current relationships
        
    Returns:
        Tuple of (updated_theorems, updated_systems, updated_relationships)
    """
    updated_theorems = existing_theorems.copy()
    updated_systems = existing_systems.copy()
    updated_relationships = existing_relationships.copy()
    
    # Track statistics
    stats = defaultdict(int)
    
    for item in scraped_data:
        # Validate data
        is_valid, message = validate_theorem_data(item)
        if not is_valid:
            stats['invalid_items'] += 1
            print(f"Skipping invalid item {item.get('title', 'Unknown')}: {message}")
            continue
        
        title = item['title']
        item_type = item['type']
        
        # Process based on type
        if item_type == 'theorem':
            if title in updated_theorems:
                # Merge with existing theorem
                updated_theorems[title] = merge_theorem_data(updated_theorems[title], item)
                stats['updated_theorems'] += 1
            else:
                # Add new theorem
                updated_theorems[title] = item
                stats['new_theorems'] += 1
        elif item_type == 'system':
            if title not in updated_systems:
                # Add new system
                updated_systems[title] = {
                    'description': item['metadata']['description'],
                    'strength': 0.5,  # Default strength
                    'contains': []  # Initialize empty containment list
                }
                stats['new_systems'] += 1
        
        # Process relationships
        existing_rel_set = {(s, t, r) for s, t, r in updated_relationships}
        for rel in item.get('relationships', []):
            source = rel['source']
            target = rel['target']
            rel_type = normalize_relationship(rel['relation_type'])
            
            rel_tuple = (source, target, rel_type)
            if rel_tuple not in existing_rel_set:
                updated_relationships.append(rel_tuple)
                stats['new_relationships'] += 1
    
    # Print integration statistics
    print("\nIntegration Statistics:")
    print(f"- New theorems added: {stats['new_theorems']}")
    print(f"- Existing theorems updated: {stats['updated_theorems']}")
    print(f"- New systems added: {stats['new_systems']}")
    print(f"- New relationships added: {stats['new_relationships']}")
    print(f"- Invalid items skipped: {stats['invalid_items']}")
    
    return updated_theorems, updated_systems, updated_relationships

def save_integrated_data(theorems: Dict[str, Any], 
                        systems: Dict[str, Any],
                        relationships: List[Tuple[str, str, str]],
                        output_dir: str):
    """Save integrated data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save theorems
    with open(os.path.join(output_dir, 'integrated_theorems.json'), 'w', encoding='utf-8') as f:
        json.dump(theorems, f, indent=2)
    
    # Save systems
    with open(os.path.join(output_dir, 'integrated_systems.json'), 'w', encoding='utf-8') as f:
        json.dump(systems, f, indent=2)
    
    # Save relationships
    with open(os.path.join(output_dir, 'integrated_relationships.json'), 'w', encoding='utf-8') as f:
        json.dump(relationships, f, indent=2)
    
    print(f"\nIntegrated data saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    from mathlogic.core.statements import (
        get_all_theorems,
        get_all_systems,
        get_all_relationships
    )
    
    # Load existing data
    existing_theorems = get_all_theorems()
    existing_systems = get_all_systems()
    existing_relationships = get_all_relationships()
    
    # Load scraped data
    scraped_data = load_scraped_data('entailment_output/scraped_data/proofwiki_data.json')
    
    if scraped_data:
        # Integrate data
        updated_theorems, updated_systems, updated_relationships = integrate_scraped_data(
            scraped_data,
            existing_theorems,
            existing_systems,
            existing_relationships
        )
        
        # Save integrated data
        save_integrated_data(
            updated_theorems,
            updated_systems,
            updated_relationships,
            'entailment_output/integrated_data'
        )
    else:
        print("No scraped data to integrate.") 