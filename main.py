import argparse
from pathlib import Path
import pandas as pd
import sys

from src.utils.logger import get_logger
from src.data_processor import DataProcessor
from src.model_pipeline import ModelPipeline
from src.visualizer import plot_tsne, plot_combined_roc_curves

logger = get_logger(__name__)

def setup_directories(base_path: Path):
    """Creates necessary directories for results if they don't exist."""
    results_dir = base_path / 'results'
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    return figures_dir, tables_dir

def main(data_path: str):
    logger.info("=== Starting Apollo Solutions ML Pipeline ===")
    
    project_root = Path(__file__).resolve().parent
    file_path = project_root / data_path
    figures_dir, tables_dir = setup_directories(project_root)
    
    if not file_path.exists():
        logger.error(f"Data file not found at {file_path}. Please check the path.")
        sys.exit(1)

    logger.info("--- Phase 1: Data Processing ---")
    processor = DataProcessor(file_path=str(file_path))
    raw_data = processor.load_data()
    df = processor.flatten_data(raw_data)
    
    logger.info("--- Phase 2: t-SNE Visualization ---")
    plot_tsne(df, save_dir=figures_dir)
    
    logger.info("PClassification & Cross-Validation ---")
    pipeline = ModelPipeline(df)
    results_df = pipeline.run_cross_validation(max_k=15, n_splits=10)
    
    logger.info("Evaluation and Metrics Export")
    euclidean_results = results_df[results_df['distance_metric'] == 'euclidean']
    cosine_results = results_df[results_df['distance_metric'] == 'cosine']
    
    best_euclidean = pipeline.find_optimal_k(euclidean_results)
    best_cosine = pipeline.find_optimal_k(cosine_results)
    
    # 6. Visualization: ROC Curves
    plot_combined_roc_curves(
        best_euclidean=best_euclidean, 
        best_cosine=best_cosine, 
        num_classes=pipeline.num_classes, 
        save_dir=figures_dir
    )

    export_df = results_df.drop(columns=['y_true_all_folds', 'y_prob_all_folds'])
    csv_path = tables_dir / 'knn_evaluation_metrics.csv'
    export_df.to_csv(csv_path, index=False)
    logger.info(f"Evaluation metrics table successfully exported to {csv_path}")
    
    logger.info("=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo Solutions ML Classification Pipeline")
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/mini_gm_public_v0.1.p',
        help='Relative path to the pickle dataset file.'
    )
    
    args = parser.parse_args()
    main(data_path=args.data)