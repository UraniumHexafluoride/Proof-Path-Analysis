import os
import shutil
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def organize_project():
    # Define main directories
    project_root = os.getcwd()
    mathlogic_dir = os.path.join(project_root, 'mathlogic')
    data_dir = os.path.join(mathlogic_dir, 'data')
    visualizations_dir = os.path.join(data_dir, 'visualizations')
    docs_dir = os.path.join(project_root, 'docs')
    archive_dir = os.path.join(project_root, 'archive')

    # Create necessary directories
    for directory in [data_dir, visualizations_dir, docs_dir, archive_dir]:
        create_directory(directory)

    # Define file mappings
    file_moves = {
        # Move CSV files to data directory
        'data': [
            'entailment_graph.csv',
            'entailment_graph_updated.csv',
            'foundational_entailment.csv',
            'proof_paths.csv',
            'axiom_dependencies.csv',
            'proof_paths_two_way.csv',
            'two_way_graph.csv',
            'bottlenecks.csv',
            'cycles_multiway.csv',
            'proof_paths_multiway.csv',
            'multiway_graph.csv',
            'cycles.csv'
        ],
        # Move visualization files
        'visualizations': [
            'entailment_key_paths.png',
            'wolfram_style_graph.png',
            'metamodeling_graph.png',
            'axiom_system_graph.png',
            'proof_space_topology.png',
            'formula_type_regions.png',
            'entailment_graph.png',
            'entailment_overview.png',
            'entailment_graph_interesting.png',
            'entailment_cone.png',
            'multiway_visual.png',
            'graph_visual.png'
        ],
        # Move documentation files
        'docs': [
            'The Road Ahead.md',
            'RESEARCH_QUESTIONS.md',
            'recommended_reading_list.txt',
            'complete_project_documentation.txt',
            'Here are my thoughts on this entail.txt',
            'Understand this uses some old code.txt',
            'full_conversation_with_code.txt',
            'Axiom Deails.docx',
            'TPHT.pdf',
            'Richard Charette Nearfinal.nb'
        ]
    }

    # Files to archive (old versions and temporary files)
    archive_files = [
        'Structural_Independence_Analysis_Ihopeisifinal.py',
        'expanded_theorems.py',
        'expanded_theorems_v2.py',
        'theorem_expansion.py',
        'entailment_theory.py',
        'expand_dataset.py',
        'data_validation.py',
        'formula_entailment.py',
        'structural_independence_analysis.py',
        'independence_results.py',
        'validate_predictions.py',
        'minimal_axiom_analysis.py',
        'expanded_entailment_data.py',
        'deep_analysis.py',
        'comprehensive_analysis.py',
        'test_file_creation.py',
        'Generate_Entailment_Graph.py',
        'enumerate_proof_paths.py',
        'relationship_discovery.py',
        'import csv.py',
        'enumerate_proof_paths - Copy.py',
        'Visualize.py',
        'Detect_Bottlenecks.py'
    ]

    # Keep these files in root
    keep_in_root = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'LICENSE',
        'research_plan.md',
        'run_analysis.py',
        'run_entailment_research.py',
        'run_minimal_axiom_analysis.py'
    ]

    # Move files to their respective directories
    for category, files in file_moves.items():
        target_dir = visualizations_dir if category == 'visualizations' else \
                    data_dir if category == 'data' else \
                    docs_dir
        
        for file in files:
            src = os.path.join(project_root, file)
            dst = os.path.join(target_dir, file)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Moved {file} to {target_dir}")

    # Move files to archive
    for file in archive_files:
        src = os.path.join(project_root, file)
        dst = os.path.join(archive_dir, file)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Archived {file}")

    print("\nProject organization complete!")
    print("\nDirectory structure:")
    print("project/")
    print("├── mathlogic/")
    print("│   ├── core/")
    print("│   ├── analysis/")
    print("│   ├── graphs/")
    print("│   ├── utils/")
    print("│   ├── tests/")
    print("│   └── data/")
    print("│       └── visualizations/")
    print("├── docs/")
    print("├── archive/")
    print("└── [root files]")

if __name__ == "__main__":
    organize_project() 