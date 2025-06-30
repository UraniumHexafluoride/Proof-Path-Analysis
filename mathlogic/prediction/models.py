"""
Predictive modeling for independence likelihood.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

class IndependencePredictorModel:
    """Model for predicting independence likelihood of mathematical statements."""
    
    def __init__(self):
        """Initialize the predictor model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, graph: nx.DiGraph, 
                        structural_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from graph structure and analysis results."""
        features = []
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') != 'theorem':
                continue
            
            node_features = {
                'node_id': node,
                'in_degree': graph.in_degree(node),
                'out_degree': graph.out_degree(node),
                'total_degree': graph.degree(node),
            }
            
            if 'centrality' in structural_results:
                centrality = structural_results['centrality'].get(node, {})
                node_features.update({
                    'betweenness': centrality.get('betweenness_centrality', 0),
                    'closeness': centrality.get('closeness_centrality', 0),
                    'pagerank': centrality.get('pagerank', 0),
                    'eigenvector': centrality.get('eigenvector_centrality', 0)
                })
            
            if 'neighborhood' in structural_results:
                neighborhood = structural_results['neighborhood'].get(node, {})
                node_features.update({
                    'neighborhood_size': neighborhood.get('size', 0),
                    'neighborhood_density': neighborhood.get('density', 0),
                    'system_neighbors': neighborhood.get('system_count', 0),
                    'theorem_neighbors': neighborhood.get('theorem_count', 0)
                })
            
            if 'strength' in structural_results:
                strength = structural_results['strength'].get(node, {})
                node_features.update({
                    'proof_power': strength.get('proof_power', 0),
                    'axiom_dependency': strength.get('axiom_dependency', 0),
                    'logical_strength': strength.get('logical_strength', 0)
                })
            
            features.append(node_features)
        
        features_df = pd.DataFrame(features)
        features_df.set_index('node_id', inplace=True)
        return features_df
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Train the independence predictor model."""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        accuracy = self.model.score(X_test_scaled, y_test)
        importances = dict(zip(features.columns, self.model.feature_importances_))
        
        return accuracy, importances
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate independence predictions for new theorems."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'confidence': np.max(probabilities, axis=1)
        }, index=features.index)
        
        return results
    
    def analyze_feature_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Analyze which features are most important for predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing importance")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'feature': features.columns[indices],
            'importance': importances[indices]
        })
        
        return importance_df
    
    def get_prediction_explanation(self, features: pd.DataFrame, 
                                 node_id: str) -> Dict[str, Any]:
        """Generate an explanation for a specific prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        node_features = features.loc[node_id]
        feature_importance = self.analyze_feature_importance(features)
        top_features = feature_importance.head(5)
        
        explanation = {
            'node_id': node_id,
            'prediction': self.predict(node_features.to_frame().T).iloc[0]['prediction'],
            'confidence': self.predict(node_features.to_frame().T).iloc[0]['confidence'],
            'top_contributing_features': top_features.to_dict('records'),
            'feature_values': node_features.to_dict()
        }
        
        return explanation
