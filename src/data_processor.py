import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Import our custom logger
from src.utils.logger import get_logger

# Initialize the logger for this specific module
logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> dict:
        """Loads the raw pickle file."""
        logger.info(f"Attempting to load data from {self.file_path}")
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info("Pickle file loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to load the pickle file. Error: {e}")
            raise

    def flatten_data(self, raw_data: dict) -> pd.DataFrame:
        """Flattens the hierarchical dictionary into a Pandas DataFrame."""
        logger.info("Flattening the data structure...")
        records = []
        
        # Iterating through the hierarchy: syndrome_id -> subject_id -> image_id
        for syndrome_id, subjects in raw_data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    
                    # Data integrity check: ensure embedding is valid and has 320 dimensions
                    if embedding is not None and len(embedding) == 320:
                        record = {
                            'syndrome_id': syndrome_id,
                            'subject_id': subject_id,
                            'image_id': image_id,
                            'embedding': np.array(embedding) # Storing as a numpy array for efficiency
                        }
                        records.append(record)
                    else:
                        logger.warning(f"Inconsistent data dropped at: {syndrome_id}/{subject_id}/{image_id}")
                        
        df = pd.DataFrame(records)
        logger.info(f"Successfully flattened {len(df)} records.")
        return df

    def get_eda_stats(self, df: pd.DataFrame) -> dict:
        """Extracts basic statistics for Exploratory Data Analysis."""
        logger.info("Extracting EDA statistics...")
        stats = {
            'total_images': len(df),
            'unique_syndromes': df['syndrome_id'].nunique(),
            'images_per_syndrome': df['syndrome_id'].value_counts().to_dict(),
            'unique_subjects': df['subject_id'].nunique()
        }
        logger.info(f"EDA Stats: {stats['unique_syndromes']} syndromes found across {stats['total_images']} images.")
        return stats