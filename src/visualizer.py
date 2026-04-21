import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from src.utils.logger import get_logger

logger = get_logger(__name__)

def plot_syndrome_distribution(df: pd.DataFrame, save_dir: Path):

    """Plots the class distribution of syndromes."""

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

def plot_embedding_magnitudes(df: pd.DataFrame, save_dir: Path):

    """Plots the distribution of L2 norms to visualize outliers."""

    logger.info("Plotting embedding magnitudes (L2 Norm)...")
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df['embedding_norm'], bins=50, kde=True, color='crimson')
    
    plt.title('Distribution of Embedding Magnitudes (Outlier Detection)', fontsize=14, pad=15)
    plt.xlabel('Vector Magnitude', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['embedding_norm'].mean(), color='black', linestyle='dashed', linewidth=2, label='Mean')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir/'eda_outliers_detection.png', dpi=300)
    plt.close()

def plot_tsne(df: pd.DataFrame, save_dir: Path):

    """Applies t-SNE to reduce embedding dimensionality to 2D and plots the result."""
    
    logger.info("starting t-SNE dimensionality reduction (this might take a few seconds)...")

    X = np.vstack(df['embedding'].values)
    y_labels = df['syndrome_id'].values
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    logger.info("t-SNE completed. Generating plot...")
    
    tsne_df = pd.DataFrame({
        'TSNE_1': X_tsne[:, 0],
        'TSNE_2': X_tsne[:, 1],
        'Syndrome': y_labels
    })
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='TSNE_1', y='TSNE_2',
        hue='Syndrome',
        palette=sns.color_palette("hsv", len(np.unique(y_labels))),
        data=tsne_df,
        legend='full',
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Syndrome Embeddings', fontsize=16, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Syndrome ID", fontsize='small')
    plt.tight_layout()
    
    plt.savefig(save_dir / 'tsne_clusters.jpg', dpi=300)
    plt.close()

def plot_combined_roc_curves(best_euclidean: dict, best_cosine: dict, num_classes: int, save_dir: Path):
    """Plots the micro average ROC curves for Euclidean and Cosine distances."""

    logger.info("Generating combined ROC curves...")
    
    plt.figure(figsize=(10, 8))
    models = {'Euclidean': best_euclidean, 'Cosine': best_cosine}
    colors = {'Euclidean': 'blue', 'Cosine': 'red'}
    
    for name, config in models.items():
        y_true = config['y_true_all_folds']
        y_prob = config['y_prob_all_folds']
        
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        plt.plot(
            fpr_micro, tpr_micro, 
            color=colors[name], lw=2, 
            label=f'{name} (k={config["k"]}) - Micro AUC: {roc_auc_micro:.4f}'
        )

    plt.plot([0, 1], [0, 1], 'k-', lw=2, label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Combined Micro-Average ROC Curve (Cross-Validation)', fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(save_dir / 'roc_curves_comparison.png', dpi=300)
    plt.close()

def plot_metrics_evolution(csv_path: Path, save_dir: Path):

    """Read the exported CSV and generates a line chart 
    elegant evolution of metrics"""

    logger.info("Generating metrics evolution")
    
    df = pd.read_csv(csv_path)
    
    # como o Cosseno é igual então o foco é na euclidiana
    df_plot = df[df['distance_metric'] == 'euclidean']
    
    plt.figure(figsize=(10, 6))
    
    # Plot F1-Score
    sns.lineplot(data=df_plot, x='k', y='avg_f1_score', 
                marker='o', color='royalblue', linewidth=2.5, label='Macro F1-Score')
    
    # Plot Top-5 Accuracy
    sns.lineplot(data=df_plot, x='k', y='avg_top5_accuracy', 
                marker='s', color='darkorange', linewidth=2.5, label='Top-5 Accuracy')
    
    plt.title('Performance of the KNN Model as a function of the K value', fontsize=15, pad=15)
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('Score (0.0 a 1.0)', fontsize=12)
    
    plt.xticks(range(1, 16)) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    save_path = save_dir / 'metrics_evolution.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Graphic saved in: {save_path}")