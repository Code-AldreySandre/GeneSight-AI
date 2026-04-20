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

def plot_tsne(df: pd.DataFrame, save_dir: Path):
    """
    Applies t-SNE to reduce embedding dimensionality to 2D and plots the result.
    """
    logger.info("Starting t-SNE dimensionality reduction (this might take a few seconds)...")

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
    
    plt.title('t-SNE Visualization of Syndrome Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    # Saving the figure
    save_path = save_dir / 'tsne_clusters.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"t-SNE plot saved successfully at {save_path}")


def plot_combined_roc_curves(best_euclidean: dict, best_cosine: dict, num_classes: int, save_dir: Path):
    """
    Plots the micro average ROC curves for both Euclidean and Cosine distances 
    on the same graph .
    """
    logger.info("Generating combined ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    models = {
        'Euclidean': best_euclidean,
        'Cosine': best_cosine
    }
    
    colors = {'Euclidean': 'blue', 'Cosine': 'red'}
    
    for name, config in models.items():
        y_true = config['y_true_all_folds']
        y_prob = config['y_prob_all_folds']
        
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        plt.plot(
            fpr_micro, 
            tpr_micro, 
            color=colors[name], 
            lw=2, 
            label=f'{name} (k={config["k"]})- Micro AUC: {roc_auc_micro:.4f}'
        )

    # plots
    plt.plot([0, 1], [0, 1], 'k-', lw=2, label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Average ROC Curve: Euclidean vs Cosine')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # saving the figure
    save_path = save_dir / 'roc_curves_comparison.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"combined ROC curve plot saved successfully at {save_path}")