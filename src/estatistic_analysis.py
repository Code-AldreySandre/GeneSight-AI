import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_eda(data_path: str, save_dir: Path):
    """Executes automated Exploratory Data Analysis and generates visual reports."""
    logger.info("--- Starting Exploratory Data Analysis (EDA) ---")
    
    processor = DataProcessor(file_path=data_path)
    raw_data = processor.load_data()
    df = processor.flatten_data(raw_data)
    
    # Ensure destination folder exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Quality Check (Missing Data)
    missing = df.isnull().sum().sum()
    logger.info(f"Data Quality: {missing} null values found in the target variables.")
    
    # 3. Class Imbalance Analysis (Syndromes)
    logger.info("Generating syndrome distribution plot...")
    plt.figure(figsize=(12, 6))
    order = df['syndrome_id'].value_counts().index
    sns.countplot(data=df, x='syndrome_id', order=order, palette='viridis')
    
    plt.title('Image Distribution per Genetic Syndrome', fontsize=14, pad=15)
    plt.xlabel('Syndrome ID', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'eda_syndrome_distribution.png', dpi=300)
    plt.close()
    
    # 4. Outlier Analysis (Embedding Magnitude)
    logger.info("Analyzing structural anomalies in embeddings (L2 Norm)...")
    df['embedding_norm'] = df['embedding'].apply(np.linalg.norm)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['embedding_norm'], bins=50, kde=True, color='crimson')
    plt.title('Distribution of Embedding Magnitudes (Outlier Detection)', fontsize=14, pad=15)
    plt.xlabel('Vector Magnitude (L2 Norm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['embedding_norm'].mean(), color='black', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'eda_outliers_detection.png', dpi=300)
    plt.close()
    
    logger.info("--- EDA completed successfully! Plots saved in 'results/figures' ---")
    return df

if __name__ == "__main__":
    # Paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "mini_gm_public_v0.1.p"
    figures_folder = project_root / "results" / "figures"
    
    run_eda(str(data_file), figures_folder)