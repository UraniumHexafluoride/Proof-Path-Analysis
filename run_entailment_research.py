"""
Main script to run the entire entailment cone research pipeline.
"""

import os
import sys
import time
import subprocess

# Output directory
OUTPUT_DIR = "entailment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_script(script_name, description):
    """Run a Python script as a subprocess and report success/failure."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    # Set environment variable for output directory
    env = os.environ.copy()
    env['ENTAILMENT_OUTPUT_DIR'] = OUTPUT_DIR
    
    # Run the script as a subprocess
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], 
                           env=env,
                           capture_output=True, 
                           text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    # Report execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Return success/failure
    return result.returncode == 0

def main():
    """Run the entailment cone research pipeline."""
    print("ENTAILMENT CONE RESEARCH PIPELINE")
    print("=================================")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"All output files will be saved to: {OUTPUT_DIR}")
    
    # Step 1: Generate the entailment graph
    run_script("Generate_Entailment_Graph.py", "Generate initial entailment graph")
    
    # Step 2: Apply logical rules
    run_script("Apply_Rules.py", "Apply logical inference rules")
    
    # Step 3: Analyze independence results
    run_script("independence_results.py", "Analyze independence results")
    
    # Step 4: Validate predictions
    run_script("validate_predictions.py", "Validate independence predictions")
    
    # Step 5: Perform structural analysis
    run_script("structural_independence_analysis.py", "Analyze structural indicators of independence")
    
    # Step 6: Analyze minimal axiom systems
    run_script("minimal_axiom_analysis.py", "Analyze minimal axiom systems")
    
    # Step 7: Analyze open problems
    run_script("analyze_open_problems.py", "Analyze open problems")
    
    # Step 8: Perform deep analysis
    run_script("deep_analysis.py", "Perform deep structural analysis")
    
    # List all generated files
    print("\nGenerated files:")
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        file_size = os.path.getsize(file_path)
        print(f"  - {filename} ({file_size} bytes)")
    
    print("\nResearch pipeline complete!")

if __name__ == "__main__":
    main()


