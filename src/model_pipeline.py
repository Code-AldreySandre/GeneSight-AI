import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger
from src.metrics import calculate_f1_score, calculate_top_k_accuracy

logger = get_logger(__name__)

class ModelPipeline:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the pipeline by extracting the features  and labels  
        from the flattened DataFrame.
        """
        logger.info("Initializing Model Pipeline...")
        self.df = df
        
        self.X = np.vstack(df['embedding'].values)
        
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(df['syndrome_id'].values)
        self.num_classes = len(self.le.classes_)
        self.groups = df['subject_id'].values
        
        logger.info(f"Feature matrix shape: {self.X.shape}")
        logger.info(f"Number of distinct syndromes (classes): {self.num_classes}")

    def run_cross_validation(self, max_k: int = 15, n_splits: int = 10):
        """
        Runs 10-fold cross-validation for K from 1 to max_k using both 
        Euclidean and Cosine distances.
        """
        logger.info(f"Starting {n_splits}-fold Cross-Validation for k=1 to {max_k}")
        
        distances = ['euclidean', 'cosine'] 
        results = []
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for distance in distances:
            logger.info(f"Evaluating Distance Metric: {distance.upper()}")
            
            for k in range(1, max_k + 1): # K de 1-15
                fold_f1 = []
                fold_top5 = []
                all_y_true = []
                all_y_prob = []

                for train_index, test_index in sgkf.split(self.X, self.y, groups=self.groups):
                    X_train, X_test = self.X[train_index], self.X[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]

                    knn = KNeighborsClassifier(n_neighbors=k, metric=distance, weights='distance')
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)

                    groups_test = self.groups[test_index]
                    y_true_subj, y_pred_subj = self.predict_by_subject(X_test, y_test, groups_test, knn)

                    y_prob = knn.predict_proba(X_test)
                    y_pred_rankings = np.argsort(y_prob, axis=1)[:, ::-1]

                    fold_f1.append(calculate_f1_score(y_true_subj, y_pred_subj))
                    fold_top5.append(calculate_top_k_accuracy(y_test, y_pred_rankings, k=5))

                    all_y_true.extend(y_test)
                    all_y_prob.extend(y_prob)
                
                avg_f1 = np.mean(fold_f1)
                avg_top5 = np.mean(fold_top5)
                logger.info(f"k={k}, {distance}; F1: {avg_f1:.4f}, Top-5 Acc: {avg_top5:.4f}")

                results.append({
                    'distance_metric': distance,
                    'k': k,
                    'avg_f1_score': avg_f1,
                    'avg_top5_accuracy': avg_top5,
                    'y_true_all_folds': np.array(all_y_true),
                    'y_prob_all_folds': np.array(all_y_prob)
                })
        return pd.DataFrame(results)

    def find_optimal_k(self, results_df: pd.DataFrame):
        """
        Identifies the optimal K based on the highest average F1-Score.
        """
        best_config = results_df.sort_values(by='avg_f1_score', ascending=False).iloc[0]
        
        logger.info("\n Optimal configuration found!")
        logger.info(f"Distance: {best_config['distance_metric']}")
        logger.info(f"K-Neighbors: {best_config['k']}")
        logger.info(f"F1-Score: {best_config['avg_f1_score']:.4f}")
        
        return best_config
    
    def predict_by_subject(self, X_test: np.ndarray, y_test: np.ndarray, groups_test: np.ndarray, knn) ->tuple:
        y_prob = knn.predict_proba(X_test)

        subject_ids = np.unique(groups_test)
        y_true_subject = []
        y_pred_subject = []

        for subject in subject_ids:
            mask = groups_test == subject
            avg_prob = y_prob[mask].mean(axis=0)
            y_pred_subject.append(np.argmax(avg_prob))
            y_true_subject.append(y_test[mask][0])

        return np.array(y_true_subject), np.array(y_pred_subject)
