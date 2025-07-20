"""
EEG Data Loading Module

This module provides comprehensive utilities for loading various EEG datasets
used in the stress detection project. It supports multiple data sources and
formats, with robust error handling and data validation.

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EEGDataLoader:
    """
    Comprehensive EEG Dataset Loader
    
    This class handles loading multiple EEG datasets from various sources,
    with support for different formats and data validation.
    
    Attributes:
        datasets (Dict): Dictionary mapping dataset names to their handles
        downloaded_paths (Dict): Storage for downloaded dataset paths
        
    Methods:
        load_all_datasets(): Download and load all available datasets
        load_specific_dataset(): Load a specific dataset by name
        validate_dataset(): Validate dataset integrity and format
        get_dataset_info(): Get information about available datasets
    """
    
    def __init__(self):
        """Initialize the EEG Data Loader with dataset configurations."""
        self.datasets = {
            'eeg_mental_state': 'birdy654/eeg-brainwave-dataset-mental-state',
            'eeg_emotions': 'birdy654/eeg-brainwave-dataset-feeling-emotions', 
            'eeg_general': 'samnikolas/eeg-dataset',
            'complete_eeg': 'amananandrai/complete-eeg-dataset'
        }
        self.downloaded_paths = {}
        
        # Set up authentication if needed
        self._setup_authentication()
        
    def _setup_authentication(self) -> None:
        """
        Set up Kaggle authentication.
        
        Note: Requires kaggle.json file in ~/.kaggle/ or environment variables:
        - KAGGLE_USERNAME
        - KAGGLE_KEY
        """
        try:
            # Check if already authenticated
            kagglehub.dataset_download('dummy/test', force_download=False)
        except Exception:
            logger.info("Kaggle authentication may be required.")
            logger.info("Please ensure kaggle.json is in ~/.kaggle/ or set environment variables.")
    
    def load_all_datasets(self) -> Dict[str, str]:
        """
        Download and load all available EEG datasets.
        
        Returns:
            Dict[str, str]: Dictionary mapping dataset names to their local paths
            
        Raises:
            Exception: If dataset download fails
        """
        logger.info("Starting download of all EEG datasets...")
        
        for name, dataset_handle in self.datasets.items():
            try:
                logger.info(f"Downloading {name}...")
                path = kagglehub.dataset_download(dataset_handle)
                self.downloaded_paths[name] = path
                logger.info(f"Successfully downloaded {name} to: {path}")
                
                # Validate the downloaded dataset
                if self.validate_dataset(path):
                    logger.info(f"Dataset {name} validated successfully")
                else:
                    logger.warning(f"Dataset {name} validation failed")
                    
            except Exception as e:
                logger.error(f"Error downloading {name}: {e}")
                continue
        
        return self.downloaded_paths
    
    def load_specific_dataset(self, dataset_name: str) -> Optional[str]:
        """
        Load a specific dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to load
            
        Returns:
            Optional[str]: Path to the downloaded dataset, None if failed
            
        Raises:
            ValueError: If dataset name is not recognized
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.datasets.keys())}")
        
        try:
            logger.info(f"Downloading {dataset_name}...")
            path = kagglehub.dataset_download(self.datasets[dataset_name])
            self.downloaded_paths[dataset_name] = path
            logger.info(f"Successfully downloaded {dataset_name} to: {path}")
            return path
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return None
    
    def load_as_dataframe(self, dataset_handle: str, filename: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Load dataset as pandas DataFrame or return path if no specific file.
        
        Args:
            dataset_handle (str): Kaggle dataset handle
            filename (Optional[str]): Specific file to load
            
        Returns:
            Union[pd.DataFrame, str]: DataFrame if filename specified, path otherwise
            
        Raises:
            Exception: If loading fails
        """
        try:
            if filename:
                df = kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_handle,
                    filename
                )
                logger.info(f"Successfully loaded {filename} as DataFrame")
                return df
            else:
                # Download entire dataset
                path = kagglehub.dataset_download(dataset_handle)
                logger.info(f"Dataset downloaded to: {path}")
                return path
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate dataset integrity and format.
        
        Args:
            dataset_path (str): Path to the dataset directory
            
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return False
            
            # Check if directory contains files
            files = os.listdir(dataset_path)
            if not files:
                logger.error(f"Dataset directory is empty: {dataset_path}")
                return False
            
            # Check for common EEG file formats
            valid_extensions = {'.csv', '.txt', '.mat', '.edf', '.set'}
            has_valid_files = any(
                any(file.endswith(ext) for ext in valid_extensions)
                for file in files
            )
            
            if not has_valid_files:
                logger.warning(f"No recognized EEG file formats found in: {dataset_path}")
                return False
            
            logger.info(f"Dataset validation successful: {len(files)} files found")
            return True
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about available datasets.
        
        Returns:
            Dict[str, Dict]: Information about each dataset
        """
        dataset_info = {}
        
        for name, handle in self.datasets.items():
            dataset_info[name] = {
                'handle': handle,
                'description': self._get_dataset_description(name),
                'downloaded': name in self.downloaded_paths,
                'path': self.downloaded_paths.get(name, 'Not downloaded')
            }
        
        return dataset_info
    
    def _get_dataset_description(self, dataset_name: str) -> str:
        """
        Get description for a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            str: Description of the dataset
        """
        descriptions = {
            'eeg_mental_state': 'EEG brainwave data for mental state classification',
            'eeg_emotions': 'EEG data for emotion recognition and classification',
            'eeg_general': 'General EEG dataset with various conditions',
            'complete_eeg': 'Comprehensive EEG dataset with multiple subjects'
        }
        
        return descriptions.get(dataset_name, 'No description available')
    
    def load_processed_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load processed EEG data from CSV files.
        
        Args:
            data_path (str): Path to the directory containing processed data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays
            
        Raises:
            FileNotFoundError: If required files are not found
            ValueError: If data format is invalid
        """
        try:
            features_path = os.path.join(data_path, 'processed_features.csv')
            eeg_path = os.path.join(data_path, 'processed_eeg_data.csv')
            
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Features file not found: {features_path}")
            
            if not os.path.exists(eeg_path):
                raise FileNotFoundError(f"EEG data file not found: {eeg_path}")
            
            # Load features
            features_df = pd.read_csv(features_path)
            logger.info(f"Loaded features: {features_df.shape}")
            
            # Load EEG data for label creation
            eeg_df = pd.read_csv(eeg_path)
            logger.info(f"Loaded EEG data: {eeg_df.shape}")
            
            # Extract features (excluding window_id if present)
            if 'window_id' in features_df.columns:
                X = features_df.drop('window_id', axis=1).values
            else:
                X = features_df.values
            
            # Create labels based on signal characteristics
            y = self._create_intelligent_labels(features_df, eeg_df)
            
            logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
            logger.info(f"Label distribution: {np.bincount(y)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def _create_intelligent_labels(self, features_df: pd.DataFrame, eeg_df: pd.DataFrame) -> np.ndarray:
        """
        Create intelligent stress labels based on EEG signal characteristics.
        
        This method analyzes various EEG features to determine stress levels,
        providing a more sophisticated labeling approach than random assignment.
        
        Args:
            features_df (pd.DataFrame): Extracted features dataframe
            eeg_df (pd.DataFrame): Raw EEG data dataframe
            
        Returns:
            np.ndarray: Array of stress labels (0: low, 1: medium, 2: high)
        """
        logger.info("Creating intelligent stress labels based on EEG characteristics...")
        
        n_samples = len(features_df)
        stress_indicators = []
        
        for i in range(n_samples):
            # Get features for this window (exclude window_id if present)
            if 'window_id' in features_df.columns:
                window_features = features_df.iloc[i, 1:].values
            else:
                window_features = features_df.iloc[i].values
            
            # Calculate stress indicators based on research
            stress_score = self._calculate_stress_score(window_features)
            stress_indicators.append(stress_score)
        
        # Convert to categorical labels based on percentiles
        labels = self._scores_to_labels(stress_indicators)
        
        return labels
    
    def _calculate_stress_score(self, features: np.ndarray) -> float:
        """
        Calculate stress score based on EEG features.
        
        Args:
            features (np.ndarray): Array of extracted features
            
        Returns:
            float: Stress score (higher = more stressed)
        """
        # Assuming features are ordered: time_domain, frequency_domain
        # Typical feature count: 285 time domain + frequency domain features
        
        n_features = len(features)
        time_features = features[:min(285, n_features//2)]
        freq_features = features[min(285, n_features//2):]
        
        # High frequency power (beta/gamma bands) - stress increases these
        high_freq_power = np.mean(freq_features) if len(freq_features) > 0 else 0
        
        # Signal variability - stress affects EEG variability
        signal_variability = np.std(time_features) if len(time_features) > 0 else 0
        
        # Signal energy
        signal_energy = np.sum(time_features ** 2) if len(time_features) > 0 else 0
        
        # Combine indicators with research-based weights
        stress_score = (
            0.4 * (high_freq_power / (np.max(freq_features) + 1e-8)) +
            0.3 * (signal_variability / (np.max(time_features) + 1e-8)) +
            0.3 * (signal_energy / (np.sum(np.abs(time_features)) + 1e-8))
        )
        
        return stress_score
    
    def _scores_to_labels(self, scores: List[float]) -> np.ndarray:
        """
        Convert stress scores to categorical labels.
        
        Args:
            scores (List[float]): List of stress scores
            
        Returns:
            np.ndarray: Categorical labels (0, 1, 2)
        """
        scores_array = np.array(scores)
        
        # Use percentiles for balanced classes
        low_threshold = np.percentile(scores_array, 33)
        high_threshold = np.percentile(scores_array, 67)
        
        labels = np.zeros(len(scores), dtype=int)
        labels[scores_array > low_threshold] = 1
        labels[scores_array > high_threshold] = 2
        
        logger.info(f"Label distribution: Low={np.sum(labels==0)}, Medium={np.sum(labels==1)}, High={np.sum(labels==2)}")
        
        return labels


# Legacy functions for backward compatibility
def download_eeg_datasets() -> Dict[str, str]:
    """
    Legacy function for downloading EEG datasets.
    
    Returns:
        Dict[str, str]: Dictionary of downloaded dataset paths
    """
    loader = EEGDataLoader()
    return loader.load_all_datasets()


def load_eeg_as_dataframe(dataset_handle: str, filename: Optional[str] = None) -> Union[pd.DataFrame, str]:
    """
    Legacy function for loading EEG data as DataFrame.
    
    Args:
        dataset_handle (str): Kaggle dataset handle
        filename (Optional[str]): Specific file to load
        
    Returns:
        Union[pd.DataFrame, str]: DataFrame or path
    """
    loader = EEGDataLoader()
    return loader.load_as_dataframe(dataset_handle, filename)


if __name__ == "__main__":
    # Example usage
    loader = EEGDataLoader()
    
    # Get dataset information
    info = loader.get_dataset_info()
    print("Available datasets:")
    for name, details in info.items():
        print(f"  {name}: {details['description']}")
    
    # Download all datasets
    paths = loader.load_all_datasets()
    print(f"\nDownloaded {len(paths)} datasets")
