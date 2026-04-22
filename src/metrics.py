import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Macro F1-Score for multiclass classification.
    
    
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Macro-average calculates the metric independently for each class and then takes the average.
    """
    classes = np.unique(y_true)
    f1_scores = []
    
    for cls in classes:
        
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        
    macro_f1 = np.mean(f1_scores)
    logger.info(f"Calculated Macro F1-Score: {macro_f1:.4f}")
    return float(macro_f1)

def calculate_top_k_accuracy(y_true: np.ndarray, y_pred_rankings: np.ndarray, k: int = 5) -> float:
    """
    Computes the Top-k Accuracy.
    
    Args:
        y_true: Array of true labels, shape (N,)
        y_pred_rankings: 2D array of predicted classes sorted by distance/probability, shape (N, num_classes)
        k: The 'k' in top-k.
        
    Returns:
        float: The proportion of times the true label is within the top k predictions.
    """
    correct_predictions = 0
    num_samples = len(y_true)
    
    for i in range(num_samples):
        if y_true[i] in y_pred_rankings[i, :k]:
            correct_predictions += 1
            
    top_k_acc = correct_predictions / num_samples
    logger.info(f"Calculated Top {k} Accuracy: {top_k_acc:.4f}")
    return float(top_k_acc)

