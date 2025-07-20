"""
Main Training Script for EEG Stress Detection Model

This script orchestrates the complete training pipeline for the CNN-LSTM
stress detection model, from data loading to model evaluation.

Usage:
    python train_model.py [--config CONFIG_FILE] [--experiment EXPERIMENT_NAME]

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import EEGDataLoader
from src.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.models.cnn_lstm_model import CNNLSTMStressDetector
from src.utils.config import (
    TRAINING_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG,
    RESULTS_DIR, MODELS_DIR, create_directories, validate_config
)
from src.utils.visualization import EEGVisualizer


def setup_logging(experiment_name: str) -> logging.Logger:
    """
    Set up logging for the training session.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        logging.Logger: Configured logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"training_{experiment_name}_{timestamp}.log"
    log_path = os.path.join(RESULTS_DIR, 'logs', log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('EEGStressTraining')
    logger.info(f"Starting training experiment: {experiment_name}")
    logger.info(f"Log file: {log_path}")
    
    return logger


def load_and_preprocess_data(logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and preprocess EEG data for training.
    
    Args:
        logger: Logger instance
        
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]: Features, labels, and metadata
    """
    logger.info("Starting data loading and preprocessing...")
    
    # Initialize data loader
    data_loader = EEGDataLoader()
    
    # Load datasets
    logger.info("Loading EEG datasets...")
    datasets = data_loader.load_all_datasets()
    
    if not datasets:
        raise ValueError("No datasets loaded successfully")
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(**PREPROCESSING_CONFIG)
    
    # Process each dataset
    all_features = []
    all_labels = []
    all_metadata = []
    
    for dataset_name, data in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Preprocess the data
            processed_data = preprocessor.preprocess_pipeline(
                data['raw_data'],
                sampling_rate=data.get('sampling_rate', 256)
            )
            
            # Extract features
            features = preprocessor.extract_all_features(processed_data)
            
            # Create labels (stress levels)
            labels = data_loader._create_intelligent_labels(
                processed_data, 
                features,
                num_classes=MODEL_CONFIG['output']['num_classes']
            )
            
            # Store results
            all_features.append(features)
            all_labels.append(labels)
            
            # Create metadata
            metadata = pd.DataFrame({
                'dataset': [dataset_name] * len(features),
                'sample_idx': range(len(features)),
                'original_shape': [str(processed_data.shape)] * len(features)
            })
            all_metadata.append(metadata)
            
            logger.info(f"Processed {len(features)} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted from any dataset")
    
    # Combine all data
    combined_features = np.vstack(all_features)
    combined_labels = np.hstack(all_labels)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    logger.info(f"Combined dataset shape: {combined_features.shape}")
    logger.info(f"Label distribution: {np.bincount(combined_labels)}")
    
    return combined_features, combined_labels, combined_metadata


def split_data(
    features: np.ndarray, 
    labels: np.ndarray, 
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        features: Feature array
        labels: Label array
        logger: Logger instance
        
    Returns:
        Tuple of train, validation, and test sets
    """
    logger.info("Splitting data into train/validation/test sets...")
    
    config = TRAINING_CONFIG['data_split']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=config['test_size'],
        stratify=labels if config['stratify'] else None,
        random_state=config['random_state']
    )
    
    # Second split: separate train and validation
    val_size_adjusted = config['validation_size'] / (1 - config['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp if config['stratify'] else None,
        random_state=config['random_state']
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    experiment_name: str,
    logger: logging.Logger
) -> CNNLSTMStressDetector:
    """
    Train the CNN-LSTM stress detection model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        experiment_name: Name of the experiment
        logger: Logger instance
        
    Returns:
        CNNLSTMStressDetector: Trained model
    """
    logger.info("Initializing CNN-LSTM model...")
    
    # Initialize model
    model = CNNLSTMStressDetector(**MODEL_CONFIG['cnn_lstm'])
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model.build_model(
        input_shape=input_shape,
        num_classes=MODEL_CONFIG['output']['num_classes']
    )
    
    logger.info(f"Model built with input shape: {input_shape}")
    logger.info(f"Model summary: {model.get_model_summary()}")
    
    # Apply data augmentation if enabled
    if TRAINING_CONFIG['augmentation']['enabled']:
        logger.info("Applying data augmentation...")
        X_train_aug, y_train_aug = model.augment_data(X_train, y_train)
        logger.info(f"Augmented training set: {X_train_aug.shape[0]} samples")
    else:
        X_train_aug, y_train_aug = X_train, y_train
    
    # Train model
    logger.info("Starting model training...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(MODELS_DIR, f"{experiment_name}_{timestamp}.h5")
    
    history = model.train(
        X_train_aug, y_train_aug,
        X_val, y_val,
        model_save_path=model_save_path,
        **TRAINING_CONFIG['training']
    )
    
    logger.info(f"Training completed. Model saved to: {model_save_path}")
    
    return model


def evaluate_model(
    model: CNNLSTMStressDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        experiment_name: Name of the experiment
        logger: Logger instance
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = model.evaluate_model(X_test, y_test)
    
    # Log metrics
    logger.info("Model Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(RESULTS_DIR, f"evaluation_{experiment_name}_{timestamp}.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Test Samples: {len(y_test)}\n\n")
        f.write("Metrics:\n")
        for metric_name, value in metrics.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
    
    logger.info(f"Detailed results saved to: {results_path}")
    
    return metrics


def create_visualizations(
    model: CNNLSTMStressDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    logger: logging.Logger
) -> None:
    """
    Create visualizations for model results.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        experiment_name: Name of the experiment
        logger: Logger instance
    """
    logger.info("Creating visualizations...")
    
    visualizer = EEGVisualizer()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Training history plots
        if hasattr(model, 'history') and model.history:
            history_path = os.path.join(
                RESULTS_DIR, 'plots', 
                f"training_history_{experiment_name}_{timestamp}.png"
            )
            visualizer.plot_training_history(model.history, save_path=history_path)
            logger.info(f"Training history saved to: {history_path}")
        
        # Confusion matrix
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm_path = os.path.join(
            RESULTS_DIR, 'plots',
            f"confusion_matrix_{experiment_name}_{timestamp}.png"
        )
        visualizer.plot_confusion_matrix(
            y_test, y_pred, 
            class_names=MODEL_CONFIG['output']['class_names'],
            save_path=cm_path
        )
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        # Performance dashboard
        dashboard_path = os.path.join(
            RESULTS_DIR, 'plots',
            f"performance_dashboard_{experiment_name}_{timestamp}.png"
        )
        
        metrics = model.evaluate_model(X_test, y_test)
        visualizer.create_performance_dashboard(
            metrics, model.history if hasattr(model, 'history') else None,
            save_path=dashboard_path
        )
        logger.info(f"Performance dashboard saved to: {dashboard_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train EEG Stress Detection Model')
    parser.add_argument('--experiment', type=str, default='default',
                        help='Name of the experiment')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Setup
    create_directories()
    validate_config()
    logger = setup_logging(args.experiment)
    
    try:
        # Load and preprocess data
        features, labels, metadata = load_and_preprocess_data(logger)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            features, labels, logger
        )
        
        # Train model
        model = train_model(
            X_train, y_train, X_val, y_val, 
            args.experiment, logger
        )
        
        # Evaluate model
        metrics = evaluate_model(
            model, X_test, y_test, 
            args.experiment, logger
        )
        
        # Create visualizations
        create_visualizations(
            model, X_test, y_test, 
            args.experiment, logger
        )
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final accuracy: {metrics.get('accuracy', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
