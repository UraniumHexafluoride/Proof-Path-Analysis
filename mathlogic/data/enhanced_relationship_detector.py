"""
Enhanced relationship detection using NLP and machine learning techniques.
"""

import spacy
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RelationshipEvidence:
    """Evidence supporting a relationship between theorems."""
    source: str
    target: str
    relationship_type: str
    confidence: float
    evidence_type: str
    supporting_text: str = ""

class EnhancedRelationshipDetector:
    """Advanced relationship detection using NLP and multiple evidence types."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.relationship_patterns = {
            'proves': [
                r'proves', r'demonstrates', r'shows', r'establishes',
                r'proof of', r'proven by', r'demonstrated by'
            ],
            'implies': [
                r'implies', r'leads to', r'results in', r'consequently',
                r'therefore', r'hence', r'thus'
            ],
            'equivalent': [
                r'equivalent to', r'if and only if', r'iff',
                r'equivalent with', r'same as', r'identical to'
            ],
            'independent': [
                r'independent of', r'not provable from', r'cannot be proved',
                r'undecidable in', r'independent from'
            ],
            'related': [
                r'related to', r'connected to', r'associated with',
                r'similar to', r'analogous to'
            ]
        }
        
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)
    
    def _extract_mathematical_concepts(self, text: str) -> Set[str]:
        """Extract mathematical concepts from text using NLP."""
        doc = self.nlp(text)
        concepts = set()
        
        # Look for mathematical terms
        math_patterns = [
            'theorem', 'lemma', 'proof', 'axiom', 'proposition',
            'corollary', 'definition', 'equation', 'formula'
        ]
        
        for token in doc:
            # Check for mathematical terms
            if any(pattern in token.text.lower() for pattern in math_patterns):
                if token.head.dep_ in ['nsubj', 'dobj', 'pobj']:
                    concepts.add(token.head.text)
            
            # Check for mathematical symbols
            if token.like_num or token.text in ['+', '-', '*', '/', '=', '≠', '≈', '∈', '∉']:
                concepts.add(token.text)
        
        return concepts
    
    def _analyze_citation_overlap(self, refs1: List[str], refs2: List[str]) -> float:
        """Compute citation overlap between two sets of references."""
        if not refs1 or not refs2:
            return 0.0
        
        set1 = set(refs1)
        set2 = set(refs2)
        overlap = len(set1.intersection(set2))
        return 2 * overlap / (len(set1) + len(set2))
    
    def detect_enhanced_relationships(self, theorems_data: Dict) -> nx.DiGraph:
        """
        Detect relationships between theorems using multiple evidence types.
        
        Args:
            theorems_data: Dictionary of theorem data including descriptions and metadata
            
        Returns:
            NetworkX DiGraph with theorems as nodes and relationships as edges
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for name, data in theorems_data.items():
            G.add_node(name, type='theorem', **data.get('metadata', {}))
        
        # Add basic relationships from metadata
        for name, data in theorems_data.items():
            metadata = data.get('metadata', {})
            
            # Add proves relationships
            for proved_by in metadata.get('proved_by', []):
                if proved_by in theorems_data:
                    G.add_edge(
                        proved_by,
                        name,
                        relationship_type='proves',
                        confidence=1.0,
                        evidence_type='metadata'
                    )
            
            # Add implies relationships
            for implies in metadata.get('implies', []):
                if implies in theorems_data:
                    G.add_edge(
                        name,
                        implies,
                        relationship_type='implies',
                        confidence=0.9,
                        evidence_type='metadata'
                    )
            
            # Add independence relationships
            for independent_from in metadata.get('independent_from', []):
                if independent_from in theorems_data:
                    G.add_edge(
                        independent_from,
                        name,
                        relationship_type='independent',
                        confidence=1.0,
                        evidence_type='metadata'
                    )
        
        # Add pattern-based relationships
        for name, data in theorems_data.items():
            desc = data['description'].lower()
            
            for other_name in theorems_data:
                if name == other_name:
                    continue
                
                other_name_lower = other_name.lower()
                if other_name_lower in desc:
                    # Check for relationship patterns
                    for rel_type, patterns in self.relationship_patterns.items():
                        for pattern in patterns:
                            if pattern in desc:
                                # Add edge if not exists or update if higher confidence
                                if not G.has_edge(name, other_name):
                                    G.add_edge(
                                        name,
                                        other_name,
                                        relationship_type=rel_type,
                                        confidence=0.8,
                                        evidence_type='pattern_match',
                                        supporting_text=desc
                                    )
                                else:
                                    edge_data = G.get_edge_data(name, other_name)
                                    if 0.8 > edge_data['confidence']:
                                        G[name][other_name].update({
                                            'relationship_type': rel_type,
                                            'confidence': 0.8,
                                            'evidence_type': 'pattern_match',
                                            'supporting_text': desc
                                        })
        
        return G
    
    def save_relationships(self, relationships: List[RelationshipEvidence], output_path: str):
        """Save detected relationships to a JSON file."""
        relationship_data = [
            {
                'source': r.source,
                'target': r.target,
                'relationship_type': r.relationship_type,
                'confidence': r.confidence,
                'evidence_type': r.evidence_type,
                'supporting_text': r.supporting_text
            }
            for r in relationships
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relationship_data, f, indent=2)
        
        logger.info(f"Saved {len(relationships)} relationships to {output_path}")
    
    def analyze_relationship_network(self, relationships: List[RelationshipEvidence]) -> Dict:
        """Analyze the network structure of detected relationships."""
        G = nx.DiGraph()
        
        # Add edges with weights based on confidence
        for rel in relationships:
            G.add_edge(
                rel.source,
                rel.target,
                weight=rel.confidence,
                relationship_type=rel.relationship_type
            )
        
        # Compute network metrics
        analysis = {
            'total_relationships': len(relationships),
            'unique_theorems': len(set(n for r in relationships for n in [r.source, r.target])),
            'relationship_types': defaultdict(int),
            'average_confidence': np.mean([r.confidence for r in relationships]),
            'evidence_type_distribution': defaultdict(int),
            'strongly_connected_components': list(nx.strongly_connected_components(G)),
            'average_clustering': nx.average_clustering(G.to_undirected()),
            'degree_centrality': nx.degree_centrality(G)
        }
        
        # Count relationship types
        for rel in relationships:
            analysis['relationship_types'][rel.relationship_type] += 1
            analysis['evidence_type_distribution'][rel.evidence_type] += 1
        
        return analysis 