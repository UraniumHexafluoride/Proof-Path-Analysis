"""
Main script to orchestrate the entire analysis workflow using chunked processing.
"""

import os
import logging
from pathlib import Path
import networkx as nx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from mathlogic.data.chunk_manager import ChunkManager
from mathlogic.data.stream_processor import StreamProcessor
from mathlogic.core.entailment import EntailmentGraph
from mathlogic.analysis.structural import StructuralAnalyzer
from mathlogic.prediction.models import IndependencePredictorModel

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """Orchestrates the entire analysis workflow using chunked processing."""
    
    def __init__(self, output_dir: str = "analysis_output"):
        """Initialize the orchestrator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.chunk_manager = ChunkManager(str(self.output_dir / "chunks"))
        self.stream_processor = StreamProcessor(str(self.output_dir / "processed"))
        self.structural_analyzer = StructuralAnalyzer()
        self.predictor_model = IndependencePredictorModel()
        
    def run_full_analysis(self, data_sources: Dict[str, Any]) -> None:
        """Run the complete analysis workflow."""
        logger.info("Starting full analysis workflow...")
        
        try:
            # Process data streams from each source
            for source_name, source_config in data_sources.items():
                logger.info(f"Processing data from source: {source_name}")
                self._process_source(source_name, source_config)
            
            # Build relationships graph
            logger.info("Building relationships graph...")
            relationships_df = self.stream_processor.build_relationships_stream()
            graph = self.stream_processor.build_graph_stream(relationships_df)
            
            # Run structural analysis
            logger.info("Running structural analysis...")
            structural_results = self._run_structural_analysis(graph)
            
            # Generate predictions
            logger.info("Generating independence predictions...")
            predictions = self._generate_predictions(graph, structural_results)
            
            # Save results
            self._save_results(structural_results, predictions)
            
            logger.info("Analysis workflow completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in analysis workflow: {str(e)}", exc_info=True)
            raise
    
    def _process_source(self, source_name: str, source_config: Dict[str, Any]) -> None:
        """Process data from a single source using chunked processing."""
        try:
            # Get data stream from source
            data_stream = source_config['stream_generator']()
            
            # Process stream with automatic chunking
            self.stream_processor.process_stream(data_stream, source_name)
            
        except Exception as e:
            logger.error(f"Error processing source {source_name}: {str(e)}")
            raise
    
    def _run_structural_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Run structural analysis on the graph using parallel processing."""
        with ThreadPoolExecutor() as executor:
            # Run different analyses in parallel
            centrality_future = executor.submit(
                self.structural_analyzer.compute_centrality_metrics, graph)
            neighborhood_future = executor.submit(
                self.structural_analyzer.analyze_neighborhood_structure, graph)
            strength_future = executor.submit(
                self.structural_analyzer.analyze_logical_strength, graph)
            
            # Collect results
            results = {
                'centrality': centrality_future.result(),
                'neighborhood': neighborhood_future.result(),
                'strength': strength_future.result()
            }
        
        return results
    
    def _generate_predictions(self, graph: nx.DiGraph, 
                            structural_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate independence predictions using the trained model."""
        # Extract features from structural results
        features = self.predictor_model.extract_features(graph, structural_results)
        
        # Generate predictions
        predictions = self.predictor_model.predict(features)
        
        return predictions
    
    def _save_results(self, structural_results: Dict[str, Any], 
                     predictions: pd.DataFrame) -> None:
        """Save analysis results to files."""
        # Save structural analysis results
        structural_path = self.output_dir / "structural_analysis.json"
        pd.DataFrame(structural_results).to_json(structural_path)
        
        # Save predictions
        predictions_path = self.output_dir / "independence_predictions.csv"
        predictions.to_csv(predictions_path, index=True)
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main entry point for the analysis workflow."""
    # Configure data sources
    data_sources = {
        'proofwiki': {
            'stream_generator': lambda: ProofWikiSource().get_stream(),
            'batch_size': 100
        },
        'arxiv': {
            'stream_generator': lambda: ArXivSource().get_stream(),
            'batch_size': 50
        },
        'wikipedia': {
            'stream_generator': lambda: WikipediaSource().get_stream(),
            'batch_size': 200
        }
    }
    
    # Run analysis
    orchestrator = AnalysisOrchestrator()
    orchestrator.run_full_analysis(data_sources)

if __name__ == "__main__":
    main()
