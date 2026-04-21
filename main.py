import argparse
from pathlib import Path
import pandas as pd
import sys

from src.utils.logger import get_logger
from src.data_processor import DataProcessor
from src.estatistic_analysis import run_eda
from src.model_pipeline import ModelPipeline
from src.visualizer import plot_tsne, plot_combined_roc_curves, plot_metrics_evolution

logger = get_logger(__name__)

def setup_directories(base_path: Path):
    results_dir = base_path / 'results'
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    return figures_dir, tables_dir

def main(data_path: str):
    logger.info("ML pipeline: Starting initialization!")
    
    project_root = Path(__file__).resolve().parent
    file_path = project_root / data_path
    figures_dir, tables_dir = setup_directories(project_root)
    
    if not file_path.exists():
        logger.error(f"Data file not found in {file_path}.")
        sys.exit(1)

    logger.info("Phase 1: Statistical Analysis and Data Quality")
    df_raw = run_eda(str(file_path), figures_dir)

    logger.info("Phase 2: Processing and L2 Normalization")
    processor = DataProcessor(file_path=str(file_path))
    raw_data = processor.load_data()
    df = processor.flatten_data(raw_data)
    
    if hasattr(processor, 'treat_embeddings'):
        df = processor.treat_embeddings(df)
        logger.info("Embedding processing completed.")
    
    logger.info("Phase 3: Cluster Visualization")
    plot_tsne(df, save_dir=figures_dir)
    
    logger.info("Phase 4: Training and Cross-Validation (10-fold)")
    pipeline = ModelPipeline(df)
    results_df = pipeline.run_cross_validation(max_k=15, n_splits=10)
    
    logger.info("Phase 5: Metrics Export")
    euclidean_results = results_df[results_df['distance_metric'] == 'euclidean']
    cosine_results = results_df[results_df['distance_metric'] == 'cosine']
    
    best_euclidean = pipeline.find_optimal_k(euclidean_results)
    best_cosine = pipeline.find_optimal_k(cosine_results)
    
    plot_combined_roc_curves(
        best_euclidean=best_euclidean, 
        best_cosine=best_cosine, 
        num_classes=pipeline.num_classes, 
        save_dir=figures_dir
    )
    
    export_df = results_df.drop(columns=['y_true_all_folds', 'y_prob_all_folds'])
    csv_path = tables_dir / 'knn_evaluation_metrics.csv'
    export_df.to_csv(csv_path, index=False)
    
    plot_metrics_evolution(csv_path, figures_dir)
    
    logger.info(f"Metrics table successfully exported to: {csv_path}")
    logger.info("Pipeline Completed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo Solutions ML Pipeline")
    parser.add_argument('--data', type=str, default='data/mini_gm_public_v0.1.p')
    args = parser.parse_args()
    main(data_path=args.data)