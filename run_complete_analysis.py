"""
Run complete analysis pipeline for mathematical independence prediction.
"""

import os
import logging
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph  # Use JSON instead of pickle for better compatibility
from datetime import datetime
from typing import Dict, List, Tuple
import json

from mathlogic.core.statements import get_all_theorems, get_all_systems, get_all_relationships
from mathlogic.data.multi_source_scraper import MultiSourceScraper
from mathlogic.data.enhanced_relationship_detector import EnhancedRelationshipDetector
from mathlogic.prediction.features import FeatureExtractor, TheoremFeatures
from mathlogic.prediction.models import IndependencePredictorModel, ModelConfig
from mathlogic.prediction.validation import ModelValidator
from mathlogic.visualization.network import NetworkVisualizer, VisualizationConfig
from mathlogic.analysis.structural import StructuralAnalyzer
from mathlogic.analysis.open_problems import OpenProblemsAnalyzer, convert_graph_to_cone
from mathlogic.data.stream_processor import StreamProcessor
from mathlogic.analysis.dependencies import build_entailment_graph, analyze_logical_strength, classify_theorems
from mathlogic.analysis.structural import analyze_centrality, analyze_neighborhood
from mathlogic.analysis.independence import analyze_structural_independence, analyze_extended_neighborhood, analyze_neighborhood_diversity
from mathlogic.graphs.visualization import visualize_entailment_graph, visualize_centrality_distribution, visualize_entailment_cone
from mathlogic.analysis.paths import find_shortest_proof_paths

# Set up logging at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

class AnalysisPipeline:
    """Complete analysis pipeline for independence prediction."""
    
    def __init__(self, output_dir: str = "entailment_output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set up file handler for this run
        self.log_file = os.path.join(self.run_dir, "analysis.log")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def collect_data(self) -> Tuple[nx.DiGraph, Dict]:
        """Collect and preprocess data from all sources."""
        logger.info("Starting data collection...")
        
        # Initialize data collection components
        scraper = MultiSourceScraper(output_dir=os.path.join(self.run_dir, 'data'))
        detector = EnhancedRelationshipDetector()
        
        try:
            # Scrape theorems from all sources
            logger.info("Scraping theorems from all sources...")
            theorems_data = scraper.scrape_all_sources()
            
            # Detect relationships
            logger.info("Detecting relationships between theorems...")
            G = detector.detect_enhanced_relationships(theorems_data)
            
            logger.info(f"Collected {len(theorems_data)} theorems and {G.number_of_edges()} relationships")
            return G, theorems_data
            
        except Exception as e:
            logger.error(f"Error during data collection: {str(e)}")
            raise
    
    def analyze_structure(self, G: nx.DiGraph) -> Dict:
        """Analyze the structural properties of the entailment graph."""
        logger.info("Starting structural analysis...")
        
        try:
            analyzer = StructuralAnalyzer()
            metrics = analyzer.analyze_graph(G)
            
            # Save results
            output_file = os.path.join(self.run_dir, 'structural_metrics.json')
            analyzer.save_metrics(metrics, output_file)
            
            logger.info("Structural analysis complete")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during structural analysis: {str(e)}")
            raise
    
    def analyze_open_problems(self, G: nx.DiGraph, metrics: Dict) -> None:
        """Analyze open problems in the graph."""
        logger.info("Starting open problems analysis...")
        
        try:
            # Convert graph to entailment cone
            cone = convert_graph_to_cone(G)
            
            # Create analyzer with cone
            analyzer = OpenProblemsAnalyzer(cone)
            results = analyzer.analyze_problems(G, metrics)
            
            # Save results
            output_file = os.path.join(self.run_dir, 'open_problems_analysis.json')
            analyzer.save_results(results, output_file)
            
            logger.info("Open problems analysis complete")
            
        except Exception as e:
            logger.error(f"Error during open problems analysis: {str(e)}")
            raise
    
    def validate_models(self, G: nx.DiGraph, metrics: Dict) -> None:
        """Validate prediction models."""
        logger.info("Starting model validation...")
        
        try:
            validator = ModelValidator(output_dir=os.path.join(self.run_dir, 'validation'))
            validation_results = validator.validate_all_models(G, metrics)
            
            logger.info("Model validation complete")
            
        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            raise
    
    def visualize_results(self, G: nx.DiGraph, metrics: Dict) -> None:
        """Create visualizations of the results."""
        logger.info("Starting visualization generation...")
        
        try:
            config = VisualizationConfig()
            visualizer = NetworkVisualizer(output_dir=os.path.join(self.run_dir, 'visualization'))
            
            # Generate visualizations
            visualizer.plot_network(G, config)
            visualizer.plot_metrics(metrics)
            
            logger.info("Visualization generation complete")
            
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            raise
    
    def run_pipeline(self) -> None:
        """Run the complete analysis pipeline."""
        try:
            # Collect and preprocess data
            G, theorems = self.collect_data()
            
            # Analyze graph structure
            metrics = self.analyze_structure(G)
            
            # Analyze open problems
            self.analyze_open_problems(G, metrics)
            
            # Validate models
            self.validate_models(G, metrics)
            
            # Create visualizations
            self.visualize_results(G, metrics)
            
            logger.info("Analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main entry point."""
    pipeline = AnalysisPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()