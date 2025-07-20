"""
Advanced CNN-LSTM Hybrid Model for EEG-Based Stress Detection

This module implements a state-of-the-art hybrid CNN-LSTM architecture for
predicting stress levels from EEG signals. The model combines convolutional
layers for local feature extraction with LSTM layers for temporal modeling.

Key Features:
- Multi-scale CNN feature extraction
- Bidirectional LSTM for temporal dependencies
- Attention mechanisms for relevant feature selection
- Advanced regularization techniques
- Data augmentation for robust training
- Comprehensive evaluation metrics

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Input, Concatenate, Attention, MultiHeadAttention,
    LayerNormalization, Reshape, Flatten, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class CNNLSTMStressDetector:
    """
    Advanced CNN-LSTM Hybrid Model for EEG Stress Detection
    
    This class implements a sophisticated deep learning architecture that combines
    Convolutional Neural Networks (CNN) for local feature extraction and Long
    Short-Term Memory (LSTM) networks for temporal pattern recognition.
    
    Architecture Features:
    - Multi-scale 1D convolutions for frequency-specific feature extraction
    - Bidirectional LSTM layers for temporal dependency modeling
    - Attention mechanisms for relevant feature selection
    - Advanced regularization (dropout, batch normalization, L2)
    - Data augmentation techniques for robustness
    
    Attributes:
        model (tf.keras.Model): The trained model
        scaler (StandardScaler): Feature scaler
        label_encoder (LabelEncoder): Label encoder for multi-class classification
        history (tf.keras.callbacks.History): Training history
        model_config (Dict): Model configuration parameters
        
    Methods:
        build_model(): Construct the CNN-LSTM architecture
        train(): Train the model with advanced techniques
        evaluate(): Comprehensive model evaluation
        predict(): Make predictions on new data
        save_model(): Save trained model and parameters
        load_model(): Load pre-trained model
    """
    
    def __init__(self, 
                 model_config: Optional[Dict] = None,
                 data_dir: str = 'data',
                 save_dir: str = 'models'):
        """
        Initialize the CNN-LSTM Stress Detector.
        
        Args:
            model_config (Optional[Dict]): Custom model configuration
            data_dir (str): Directory containing training data
            save_dir (str): Directory for saving models and results
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # Create directories if they don't exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # Default model configuration
        self.model_config = model_config or {
            'cnn_filters': [32, 64, 128],
            'cnn_kernels': [3, 5, 7],
            'lstm_units': [64, 32],
            'dense_units': [128, 64],
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.3,
            'l2_reg': 0.001,
            'learning_rate': 0.001,
            'num_classes': 3
        }
        
        logger.info("CNN-LSTM Stress Detector initialized")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> Model:
        """
        Build the advanced CNN-LSTM hybrid architecture.
        
        This method constructs a sophisticated model that combines:
        1. Multi-scale CNN for local feature extraction
        2. Bidirectional LSTM for temporal modeling
        3. Attention mechanism for feature selection
        4. Dense layers for final classification
        
        Args:
            input_shape (Tuple[int, ...]): Shape of input data (timesteps, features)
            
        Returns:
            Model: Compiled Keras model
        """
        logger.info(f"Building CNN-LSTM model with input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape, name='eeg_input')
        
        # CNN Branch for Local Feature Extraction
        cnn_layers = []
        for i, (filters, kernel_size) in enumerate(zip(
            self.model_config['cnn_filters'], 
            self.model_config['cnn_kernels']
        )):
            if i == 0:
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_{i+1}'
                )(inputs)
            else:
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_{i+1}'
                )(x)
            
            x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = Dropout(self.model_config['dropout_rate'], name=f'dropout_conv_{i+1}')(x)
            
            cnn_layers.append(x)
        
        # Use the final CNN layer for further processing
        cnn_output = cnn_layers[-1]
        
        # LSTM Branch for Temporal Modeling
        lstm_input = inputs  # Use original input for LSTM
        
        for i, units in enumerate(self.model_config['lstm_units']):
            return_sequences = i < len(self.model_config['lstm_units']) - 1
            
            lstm_input = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['recurrent_dropout'],
                    kernel_regularizer=l2(self.model_config['l2_reg']),
                    name=f'lstm_{i+1}'
                ),
                name=f'bidirectional_lstm_{i+1}'
            )(lstm_input)
            
            if return_sequences:
                lstm_input = BatchNormalization(name=f'bn_lstm_{i+1}')(lstm_input)
        
        lstm_output = lstm_input
        
        # Attention Mechanism (if we have sequential data from CNN)
        if len(cnn_output.shape) == 3:  # (batch, timesteps, features)
            try:
                attention_output = MultiHeadAttention(
                    num_heads=4,
                    key_dim=32,
                    name='multi_head_attention'
                )(cnn_output, cnn_output)
                
                attention_output = LayerNormalization(name='attention_norm')(attention_output)
                cnn_pooled = GlobalAveragePooling1D(name='attention_pooling')(attention_output)
            except:
                # Fallback to simple pooling if attention fails
                cnn_pooled = GlobalAveragePooling1D(name='cnn_pooling')(cnn_output)
        else:
            cnn_pooled = cnn_output
        
        # Ensure LSTM output is properly shaped
        if len(lstm_output.shape) > 2:
            lstm_pooled = GlobalAveragePooling1D(name='lstm_pooling')(lstm_output)
        else:
            lstm_pooled = lstm_output
        
        # Combine CNN and LSTM features
        try:
            combined = Concatenate(name='feature_fusion')([cnn_pooled, lstm_pooled])
        except:
            # If concatenation fails, use CNN features only
            logger.warning("Feature concatenation failed, using CNN features only")
            combined = cnn_pooled
        
        # Dense layers for classification
        x = combined
        for i, units in enumerate(self.model_config['dense_units']):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(self.model_config['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        if self.model_config['num_classes'] > 2:
            # Multi-class classification
            outputs = Dense(
                self.model_config['num_classes'],
                activation='softmax',
                name='stress_output'
            )(x)
            loss = 'sparse_categorical_crossentropy'
        else:
            # Binary classification
            outputs = Dense(
                1,
                activation='sigmoid',
                name='stress_output'
            )(x)
            loss = 'binary_crossentropy'
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_StressDetector')
        
        # Compile model
        optimizer = Adam(
            learning_rate=self.model_config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Print model summary
        model.summary()
        
        logger.info("Model built and compiled successfully")
        return model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    validation_split: float = 0.2,
                    augment_data: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training with preprocessing and augmentation.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            test_size (float): Proportion of data for testing
            validation_split (float): Proportion of training data for validation
            augment_data (bool): Whether to apply data augmentation
            
        Returns:
            Tuple[np.ndarray, ...]: Prepared training and testing sets
        """
        logger.info("Preparing data for training...")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply data augmentation to training set only
        if augment_data:
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            X_train = X_train_aug
            y_train = y_train_aug
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(X_train.shape[0], -1)
        ).reshape(X_train.shape)
        
        X_test_scaled = self.scaler.transform(
            X_test.reshape(X_test.shape[0], -1)
        ).reshape(X_test.shape)
        
        # Encode labels if needed
        if len(np.unique(y)) > 2:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
        
        # Reshape for CNN-LSTM if needed
        if len(X_train_scaled.shape) == 2:
            # Reshape to (samples, timesteps, features)
            timesteps = min(100, X_train_scaled.shape[1] // 10)
            features = X_train_scaled.shape[1] // timesteps
            
            if X_train_scaled.shape[1] % timesteps != 0:
                # Pad to make it divisible
                pad_size = timesteps - (X_train_scaled.shape[1] % timesteps)
                X_train_scaled = np.pad(X_train_scaled, ((0, 0), (0, pad_size)), mode='constant')
                X_test_scaled = np.pad(X_test_scaled, ((0, 0), (0, pad_size)), mode='constant')
                features = X_train_scaled.shape[1] // timesteps
            
            X_train_reshaped = X_train_scaled.reshape(-1, timesteps, features)
            X_test_reshaped = X_test_scaled.reshape(-1, timesteps, features)
        else:
            X_train_reshaped = X_train_scaled
            X_test_reshaped = X_test_scaled
        
        logger.info(f"Data prepared: Train={X_train_reshaped.shape}, Test={X_test_reshaped.shape}")
        
        return X_train_reshaped, X_test_reshaped, y_train_encoded, y_test_encoded
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply advanced data augmentation techniques for EEG signals.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Labels
            augment_factor (int): Number of augmented samples per original
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented data and labels
        """
        logger.info(f"Applying data augmentation (factor: {augment_factor})...")
        
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(augment_factor):
            X_temp = X.copy()
            
            # 1. Gaussian noise injection
            noise_factor = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_factor, X_temp.shape)
            X_temp += noise
            
            # 2. Amplitude scaling
            scale_factors = np.random.uniform(0.8, 1.2, X_temp.shape[0])
            for i in range(X_temp.shape[0]):
                X_temp[i] *= scale_factors[i]
            
            # 3. Time shifting (if data is sequential)
            if len(X_temp.shape) == 3:  # (samples, timesteps, features)
                for i in range(X_temp.shape[0]):
                    shift = np.random.randint(-X_temp.shape[1]//4, X_temp.shape[1]//4)
                    X_temp[i] = np.roll(X_temp[i], shift, axis=0)
            
            # 4. Feature dropout
            dropout_prob = 0.05
            dropout_mask = np.random.random(X_temp.shape) > dropout_prob
            X_temp *= dropout_mask
            
            X_aug.append(X_temp)
            y_aug.append(y)
        
        X_augmented = np.vstack(X_aug)
        y_augmented = np.hstack(y_aug)
        
        logger.info(f"Data augmented: {X.shape} -> {X_augmented.shape}")
        return X_augmented, y_augmented
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.2,
             **kwargs) -> Any:
        """
        Train the CNN-LSTM model with advanced techniques.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion for validation
            **kwargs: Additional training parameters
            
        Returns:
            History: Training history object
        """
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X_train.shape[1:])
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Define callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate on test set
        test_results = self.evaluate(X_test, y_test)
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_results['f1_score']:.4f}")
        
        # Save final model
        self.save_model()
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X, verbose=0)
        
        if self.model_config['num_classes'] > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        
        # Print detailed results
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Classification report
        if self.model_config['num_classes'] > 2:
            target_names = ['Low Stress', 'Medium Stress', 'High Stress']
        else:
            target_names = ['No Stress', 'Stress']
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        self.plot_confusion_matrix(cm, target_names)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self) -> None:
        """
        Plot comprehensive training history.
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, target_names: List[str]) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            target_names (List[str]): Class names
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - EEG Stress Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_scaled, verbose=0)
        
        if self.model_config['num_classes'] > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        return y_pred, y_pred_proba
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained model and associated parameters.
        
        Args:
            filepath (Optional[str]): Custom filepath for saving
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f'cnn_lstm_stress_model_{timestamp}.h5')
        
        self.model.save(filepath)
        
        # Save additional parameters
        import pickle
        params_path = filepath.replace('.h5', '_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_config': self.model_config
            }, f)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Parameters saved to {params_path}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model and associated parameters.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        
        # Load additional parameters
        import pickle
        params_path = filepath.replace('.h5', '_params.pkl')
        
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                self.scaler = params['scaler']
                self.label_encoder = params['label_encoder']
                self.model_config = params['model_config']
        
        logger.info(f"Model loaded from {filepath}")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5) -> List[float]:
        """
        Perform cross-validation for robust model evaluation.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            List[float]: Cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold + 1}/{cv_folds}")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Create and train model for this fold
            model_cv = self.build_model(X_train_cv.shape[1:])
            
            # Train with fewer epochs for CV
            model_cv.fit(
                X_train_cv, y_train_cv,
                validation_data=(X_val_cv, y_val_cv),
                epochs=30,
                batch_size=16,
                verbose=0
            )
            
            # Evaluate
            y_pred_cv = model_cv.predict(X_val_cv, verbose=0)
            if self.model_config['num_classes'] > 2:
                y_pred_cv = np.argmax(y_pred_cv, axis=1)
            else:
                y_pred_cv = (y_pred_cv > 0.5).astype(int).flatten()
            
            accuracy_cv = accuracy_score(y_val_cv, y_pred_cv)
            cv_scores.append(accuracy_cv)
            
            logger.info(f"Fold {fold + 1} Accuracy: {accuracy_cv:.4f}")
        
        logger.info(f"CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        return cv_scores


if __name__ == "__main__":
    # Example usage
    detector = CNNLSTMStressDetector()
    
    # Generate sample data for testing
    n_samples, n_features = 1000, 100
    X_sample = np.random.randn(n_samples, n_features)
    y_sample = np.random.randint(0, 3, n_samples)
    
    # Train model
    history = detector.train(X_sample, y_sample, epochs=5)
    
    # Evaluate model
    results = detector.evaluate(X_sample[:200], y_sample[:200])
    
    print("Training completed successfully!")
