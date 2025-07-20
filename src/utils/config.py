"""
Configuration Settings for EEG Stress Detection Project

This module contains all configuration parameters and constants used throughout
the project. It provides centralized configuration management for easy
parameter tuning and experimentation.

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import os
from typing import Dict, List, Tuple, Any

# Project Information
PROJECT_NAME = "Predicting Stress Peaks based on EEG Signals by Hybrid Approach of CNN-LSTM"
VERSION = "1.0.0"
AUTHORS = ["Youssef Mahdi", "Hamza El Haiki"]

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Dataset Configuration
DATASET_CONFIG = {
    'datasets': {
        'eeg_mental_state': 'birdy654/eeg-brainwave-dataset-mental-state',
        'eeg_emotions': 'birdy654/eeg-brainwave-dataset-feeling-emotions', 
        'eeg_general': 'samnikolas/eeg-dataset',
        'complete_eeg': 'amananandrai/complete-eeg-dataset'
    },
    'primary_dataset': 'eeg_mental_state',
    'sampling_rates': {
        'original': 256,
        'target': 128
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'filtering': {
        'bandpass_low': 0.5,
        'bandpass_high': 50.0,
        'notch_freq': 50.0,
        'notch_quality': 30.0,
        'filter_order': 4
    },
    'ica': {
        'n_components': 15,
        'artifact_threshold': 3.0,
        'max_iter': 1000,
        'tolerance': 1e-4
    },
    'feature_extraction': {
        'window_size': 512,
        'overlap': 256,
        'frequency_bands': {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    },
    'normalization': {
        'method': 'standard',  # 'standard', 'robust', 'minmax'
        'per_channel': True
    }
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'cnn_lstm': {
        'cnn_filters': [32, 64, 128],
        'cnn_kernels': [3, 5, 7],
        'lstm_units': [64, 32],
        'dense_units': [128, 64],
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.3,
        'l2_reg': 0.001,
        'attention_heads': 4,
        'attention_key_dim': 32
    },
    'output': {
        'num_classes': 3,  # Low, Medium, High stress
        'class_names': ['Low Stress', 'Medium Stress', 'High Stress'],
        'activation': 'softmax'  # 'softmax' for multi-class, 'sigmoid' for binary
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'data_split': {
        'train_size': 0.7,
        'validation_size': 0.15,
        'test_size': 0.15,
        'stratify': True,
        'random_state': 42
    },
    'augmentation': {
        'enabled': True,
        'augment_factor': 3,
        'noise_factor': (0.01, 0.05),
        'scale_factor': (0.8, 1.2),
        'time_shift_factor': 0.25,
        'dropout_prob': 0.05
    },
    'optimization': {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-7,
        'loss_function': 'sparse_categorical_crossentropy'
    },
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2,
        'shuffle': True,
        'verbose': 1
    },
    'callbacks': {
        'early_stopping': {
            'monitor': 'val_accuracy',
            'patience': 20,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-7
        },
        'model_checkpoint': {
            'monitor': 'val_accuracy',
            'save_best_only': True,
            'save_weights_only': False
        }
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score'
    ],
    'cross_validation': {
        'enabled': True,
        'n_folds': 5,
        'shuffle': True,
        'random_state': 42
    },
    'confusion_matrix': {
        'normalize': False,
        'display_labels': True
    }
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'plot_settings': {
        'figure_size': (12, 8),
        'dpi': 300,
        'style': 'seaborn-v0_8',
        'color_palette': 'husl',
        'save_format': 'png'
    },
    'eeg_plot': {
        'max_channels': 8,
        'default_duration': 4.0,  # seconds
        'line_width': 0.8,
        'alpha': 0.8
    },
    'feature_plot': {
        'max_features': 16,
        'bins': 20,
        'alpha': 0.6
    },
    'correlation': {
        'max_features': 50,
        'colormap': 'coolwarm',
        'mask_upper': True
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': True,
        'file': True,
        'file_path': os.path.join(RESULTS_DIR, 'logs', 'eeg_stress_detection.log')
    }
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
    'mixed_precision': False,
    'random_seed': 42
}

# File Naming Conventions
FILE_NAMING = {
    'timestamp_format': '%Y%m%d_%H%M%S',
    'model_prefix': 'cnn_lstm_stress_model',
    'results_prefix': 'results',
    'plot_prefix': 'plot'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'minimum_accuracy': 0.75,
    'good_accuracy': 0.85,
    'excellent_accuracy': 0.90,
    'minimum_f1_score': 0.70,
    'good_f1_score': 0.80
}

# EEG Signal Properties
EEG_PROPERTIES = {
    'channels': {
        'standard_10_20': [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz'
        ],
        'extended': [
            'AF3', 'AF4', 'FC1', 'FC2', 'CP1', 'CP2',
            'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
        ]
    },
    'artifacts': {
        'eog': 'Electrooculogram (eye movements)',
        'emg': 'Electromyogram (muscle activity)',
        'ecg': 'Electrocardiogram (heart activity)',
        'powerline': 'Power line interference (50/60 Hz)'
    },
    'frequency_bands': {
        'delta': {'range': (0.5, 4), 'description': 'Deep sleep, unconscious'},
        'theta': {'range': (4, 8), 'description': 'Drowsiness, meditation'},
        'alpha': {'range': (8, 13), 'description': 'Relaxed, eyes closed'},
        'beta': {'range': (13, 30), 'description': 'Alert, active thinking'},
        'gamma': {'range': (30, 100), 'description': 'High cognitive functions'}
    }
}

# Stress Level Definitions
STRESS_DEFINITIONS = {
    0: {
        'name': 'Low Stress',
        'description': 'Relaxed state, minimal physiological arousal',
        'characteristics': ['Low heart rate', 'Relaxed muscles', 'Calm breathing']
    },
    1: {
        'name': 'Medium Stress',
        'description': 'Moderate arousal, manageable stress levels',
        'characteristics': ['Slightly elevated heart rate', 'Some tension', 'Active attention']
    },
    2: {
        'name': 'High Stress',
        'description': 'High arousal, significant stress response',
        'characteristics': ['Elevated heart rate', 'Muscle tension', 'Rapid breathing']
    }
}

# Default Hyperparameters for Grid Search
HYPERPARAMETER_GRID = {
    'cnn_filters': [[16, 32, 64], [32, 64, 128], [64, 128, 256]],
    'lstm_units': [[32, 16], [64, 32], [128, 64]],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64]
}

# API Configuration (for future web interface)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'allowed_extensions': ['.csv', '.txt', '.edf', '.mat']
}


def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get a specific configuration dictionary.
    
    Args:
        config_name (str): Name of the configuration to retrieve
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If configuration name is not found
    """
    configs = {
        'dataset': DATASET_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'eeg': EEG_PROPERTIES,
        'stress': STRESS_DEFINITIONS
    }
    
    if config_name not in configs:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(configs.keys())}")
    
    return configs[config_name]


def create_directories() -> None:
    """Create all necessary directories for the project."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
        os.path.join(RESULTS_DIR, 'logs'),
        os.path.join(MODELS_DIR, 'saved_models'),
        os.path.join(MODELS_DIR, 'checkpoints')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_config() -> bool:
    """
    Validate configuration settings for consistency.
    
    Returns:
        bool: True if configuration is valid
    """
    # Check data split proportions
    split_sum = (TRAINING_CONFIG['data_split']['train_size'] + 
                TRAINING_CONFIG['data_split']['validation_size'] + 
                TRAINING_CONFIG['data_split']['test_size'])
    
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"Data split proportions must sum to 1.0, got {split_sum}")
    
    # Check frequency bands
    bands = PREPROCESSING_CONFIG['feature_extraction']['frequency_bands']
    for band_name, (low, high) in bands.items():
        if low >= high:
            raise ValueError(f"Invalid frequency band {band_name}: {low} >= {high}")
    
    # Check model parameters
    if MODEL_CONFIG['output']['num_classes'] < 2:
        raise ValueError("Number of classes must be at least 2")
    
    return True


if __name__ == "__main__":
    # Test configuration
    validate_config()
    create_directories()
    
    print(f"Project: {PROJECT_NAME}")
    print(f"Version: {VERSION}")
    print(f"Authors: {', '.join(AUTHORS)}")
    print("\nConfiguration validation passed!")
    print("All directories created successfully!")
