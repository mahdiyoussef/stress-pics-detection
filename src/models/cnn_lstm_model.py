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
        
        lstm_output = lstm_input\n        \n        # Attention Mechanism (if we have sequential data from CNN)\n        if len(cnn_output.shape) == 3:  # (batch, timesteps, features)\n            try:\n                attention_output = MultiHeadAttention(\n                    num_heads=4,\n                    key_dim=32,\n                    name='multi_head_attention'\n                )(cnn_output, cnn_output)\n                \n                attention_output = LayerNormalization(name='attention_norm')(attention_output)\n                cnn_pooled = GlobalAveragePooling1D(name='attention_pooling')(attention_output)\n            except:\n                # Fallback to simple pooling if attention fails\n                cnn_pooled = GlobalAveragePooling1D(name='cnn_pooling')(cnn_output)\n        else:\n            cnn_pooled = cnn_output\n        \n        # Ensure LSTM output is properly shaped\n        if len(lstm_output.shape) > 2:\n            lstm_pooled = GlobalAveragePooling1D(name='lstm_pooling')(lstm_output)\n        else:\n            lstm_pooled = lstm_output\n        \n        # Combine CNN and LSTM features\n        try:\n            combined = Concatenate(name='feature_fusion')([cnn_pooled, lstm_pooled])\n        except:\n            # If concatenation fails, use CNN features only\n            logger.warning(\"Feature concatenation failed, using CNN features only\")\n            combined = cnn_pooled\n        \n        # Dense layers for classification\n        x = combined\n        for i, units in enumerate(self.model_config['dense_units']):\n            x = Dense(\n                units=units,\n                activation='relu',\n                kernel_regularizer=l2(self.model_config['l2_reg']),\n                name=f'dense_{i+1}'\n            )(x)\n            x = BatchNormalization(name=f'bn_dense_{i+1}')(x)\n            x = Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)\n        \n        # Output layer\n        if self.model_config['num_classes'] > 2:\n            # Multi-class classification\n            outputs = Dense(\n                self.model_config['num_classes'],\n                activation='softmax',\n                name='stress_output'\n            )(x)\n            loss = 'sparse_categorical_crossentropy'\n        else:\n            # Binary classification\n            outputs = Dense(\n                1,\n                activation='sigmoid',\n                name='stress_output'\n            )(x)\n            loss = 'binary_crossentropy'\n        \n        # Create model\n        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_StressDetector')\n        \n        # Compile model\n        optimizer = Adam(\n            learning_rate=self.model_config['learning_rate'],\n            beta_1=0.9,\n            beta_2=0.999,\n            epsilon=1e-7\n        )\n        \n        model.compile(\n            optimizer=optimizer,\n            loss=loss,\n            metrics=['accuracy', 'precision', 'recall']\n        )\n        \n        # Print model summary\n        model.summary()\n        \n        logger.info(\"Model built and compiled successfully\")\n        return model\n    \n    def prepare_data(self, X: np.ndarray, y: np.ndarray, \n                    test_size: float = 0.2,\n                    validation_split: float = 0.2,\n                    augment_data: bool = True) -> Tuple[np.ndarray, ...]:\n        \"\"\"\n        Prepare data for training with preprocessing and augmentation.\n        \n        Args:\n            X (np.ndarray): Input features\n            y (np.ndarray): Target labels\n            test_size (float): Proportion of data for testing\n            validation_split (float): Proportion of training data for validation\n            augment_data (bool): Whether to apply data augmentation\n            \n        Returns:\n            Tuple[np.ndarray, ...]: Prepared training and testing sets\n        \"\"\"\n        logger.info(\"Preparing data for training...\")\n        \n        # Split data into train and test sets\n        X_train, X_test, y_train, y_test = train_test_split(\n            X, y, test_size=test_size, random_state=42, stratify=y\n        )\n        \n        # Apply data augmentation to training set only\n        if augment_data:\n            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)\n            X_train = X_train_aug\n            y_train = y_train_aug\n        \n        # Normalize features\n        X_train_scaled = self.scaler.fit_transform(\n            X_train.reshape(X_train.shape[0], -1)\n        ).reshape(X_train.shape)\n        \n        X_test_scaled = self.scaler.transform(\n            X_test.reshape(X_test.shape[0], -1)\n        ).reshape(X_test.shape)\n        \n        # Encode labels if needed\n        if len(np.unique(y)) > 2:\n            y_train_encoded = self.label_encoder.fit_transform(y_train)\n            y_test_encoded = self.label_encoder.transform(y_test)\n        else:\n            y_train_encoded = y_train\n            y_test_encoded = y_test\n        \n        # Reshape for CNN-LSTM if needed\n        if len(X_train_scaled.shape) == 2:\n            # Reshape to (samples, timesteps, features)\n            timesteps = min(100, X_train_scaled.shape[1] // 10)\n            features = X_train_scaled.shape[1] // timesteps\n            \n            if X_train_scaled.shape[1] % timesteps != 0:\n                # Pad to make it divisible\n                pad_size = timesteps - (X_train_scaled.shape[1] % timesteps)\n                X_train_scaled = np.pad(X_train_scaled, ((0, 0), (0, pad_size)), mode='constant')\n                X_test_scaled = np.pad(X_test_scaled, ((0, 0), (0, pad_size)), mode='constant')\n                features = X_train_scaled.shape[1] // timesteps\n            \n            X_train_reshaped = X_train_scaled.reshape(-1, timesteps, features)\n            X_test_reshaped = X_test_scaled.reshape(-1, timesteps, features)\n        else:\n            X_train_reshaped = X_train_scaled\n            X_test_reshaped = X_test_scaled\n        \n        logger.info(f\"Data prepared: Train={X_train_reshaped.shape}, Test={X_test_reshaped.shape}\")\n        \n        return X_train_reshaped, X_test_reshaped, y_train_encoded, y_test_encoded\n    \n    def augment_data(self, X: np.ndarray, y: np.ndarray, \n                    augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:\n        \"\"\"\n        Apply advanced data augmentation techniques for EEG signals.\n        \n        Args:\n            X (np.ndarray): Input data\n            y (np.ndarray): Labels\n            augment_factor (int): Number of augmented samples per original\n            \n        Returns:\n            Tuple[np.ndarray, np.ndarray]: Augmented data and labels\n        \"\"\"\n        logger.info(f\"Applying data augmentation (factor: {augment_factor})...\")\n        \n        X_aug = [X]\n        y_aug = [y]\n        \n        for _ in range(augment_factor):\n            X_temp = X.copy()\n            \n            # 1. Gaussian noise injection\n            noise_factor = np.random.uniform(0.01, 0.05)\n            noise = np.random.normal(0, noise_factor, X_temp.shape)\n            X_temp += noise\n            \n            # 2. Amplitude scaling\n            scale_factors = np.random.uniform(0.8, 1.2, X_temp.shape[0])\n            for i in range(X_temp.shape[0]):\n                X_temp[i] *= scale_factors[i]\n            \n            # 3. Time shifting (if data is sequential)\n            if len(X_temp.shape) == 3:  # (samples, timesteps, features)\n                for i in range(X_temp.shape[0]):\n                    shift = np.random.randint(-X_temp.shape[1]//4, X_temp.shape[1]//4)\n                    X_temp[i] = np.roll(X_temp[i], shift, axis=0)\n            \n            # 4. Feature dropout\n            dropout_prob = 0.05\n            dropout_mask = np.random.random(X_temp.shape) > dropout_prob\n            X_temp *= dropout_mask\n            \n            X_aug.append(X_temp)\n            y_aug.append(y)\n        \n        X_augmented = np.vstack(X_aug)\n        y_augmented = np.hstack(y_aug)\n        \n        logger.info(f\"Data augmented: {X.shape} -> {X_augmented.shape}\")\n        return X_augmented, y_augmented\n    \n    def train(self, X: np.ndarray, y: np.ndarray, \n             epochs: int = 100,\n             batch_size: int = 32,\n             validation_split: float = 0.2,\n             **kwargs) -> Any:\n        \"\"\"\n        Train the CNN-LSTM model with advanced techniques.\n        \n        Args:\n            X (np.ndarray): Training features\n            y (np.ndarray): Training labels\n            epochs (int): Number of training epochs\n            batch_size (int): Batch size for training\n            validation_split (float): Proportion for validation\n            **kwargs: Additional training parameters\n            \n        Returns:\n            History: Training history object\n        \"\"\"\n        logger.info(\"Starting model training...\")\n        \n        # Prepare data\n        X_train, X_test, y_train, y_test = self.prepare_data(X, y)\n        \n        # Build model if not already built\n        if self.model is None:\n            self.model = self.build_model(X_train.shape[1:])\n        \n        # Calculate class weights for imbalanced data\n        class_weights = compute_class_weight(\n            'balanced',\n            classes=np.unique(y_train),\n            y=y_train\n        )\n        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}\n        \n        logger.info(f\"Class weights: {class_weight_dict}\")\n        \n        # Define callbacks\n        callbacks_list = [\n            EarlyStopping(\n                monitor='val_accuracy',\n                patience=20,\n                restore_best_weights=True,\n                verbose=1\n            ),\n            ReduceLROnPlateau(\n                monitor='val_loss',\n                factor=0.5,\n                patience=10,\n                min_lr=1e-7,\n                verbose=1\n            ),\n            ModelCheckpoint(\n                filepath=os.path.join(self.save_dir, 'best_model.h5'),\n                monitor='val_accuracy',\n                save_best_only=True,\n                verbose=1\n            )\n        ]\n        \n        # Train model\n        self.history = self.model.fit(\n            X_train, y_train,\n            validation_split=validation_split,\n            epochs=epochs,\n            batch_size=batch_size,\n            class_weight=class_weight_dict,\n            callbacks=callbacks_list,\n            verbose=1\n        )\n        \n        # Evaluate on test set\n        test_results = self.evaluate(X_test, y_test)\n        logger.info(f\"Test Accuracy: {test_results['accuracy']:.4f}\")\n        logger.info(f\"Test F1-Score: {test_results['f1_score']:.4f}\")\n        \n        # Save final model\n        self.save_model()\n        \n        return self.history\n    \n    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:\n        \"\"\"\n        Comprehensive model evaluation.\n        \n        Args:\n            X (np.ndarray): Test features\n            y (np.ndarray): Test labels\n            \n        Returns:\n            Dict[str, float]: Evaluation metrics\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not trained yet. Call train() first.\")\n        \n        logger.info(\"Evaluating model...\")\n        \n        # Make predictions\n        y_pred_proba = self.model.predict(X, verbose=0)\n        \n        if self.model_config['num_classes'] > 2:\n            y_pred = np.argmax(y_pred_proba, axis=1)\n        else:\n            y_pred = (y_pred_proba > 0.5).astype(int).flatten()\n        \n        # Calculate metrics\n        accuracy = accuracy_score(y, y_pred)\n        f1 = f1_score(y, y_pred, average='weighted')\n        precision = precision_score(y, y_pred, average='weighted')\n        recall = recall_score(y, y_pred, average='weighted')\n        \n        # Print detailed results\n        print(\"\\n=== Model Evaluation Results ===\")\n        print(f\"Accuracy: {accuracy:.4f}\")\n        print(f\"F1-Score: {f1:.4f}\")\n        print(f\"Precision: {precision:.4f}\")\n        print(f\"Recall: {recall:.4f}\")\n        \n        # Classification report\n        if self.model_config['num_classes'] > 2:\n            target_names = ['Low Stress', 'Medium Stress', 'High Stress']\n        else:\n            target_names = ['No Stress', 'Stress']\n        \n        print(\"\\nClassification Report:\")\n        print(classification_report(y, y_pred, target_names=target_names))\n        \n        # Confusion matrix\n        cm = confusion_matrix(y, y_pred)\n        self.plot_confusion_matrix(cm, target_names)\n        \n        return {\n            'accuracy': accuracy,\n            'f1_score': f1,\n            'precision': precision,\n            'recall': recall,\n            'confusion_matrix': cm\n        }\n    \n    def plot_training_history(self) -> None:\n        \"\"\"\n        Plot comprehensive training history.\n        \"\"\"\n        if self.history is None:\n            logger.warning(\"No training history available\")\n            return\n        \n        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n        \n        # Accuracy\n        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)\n        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)\n        axes[0, 0].set_title('Model Accuracy')\n        axes[0, 0].set_xlabel('Epoch')\n        axes[0, 0].set_ylabel('Accuracy')\n        axes[0, 0].legend()\n        axes[0, 0].grid(True, alpha=0.3)\n        \n        # Loss\n        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)\n        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)\n        axes[0, 1].set_title('Model Loss')\n        axes[0, 1].set_xlabel('Epoch')\n        axes[0, 1].set_ylabel('Loss')\n        axes[0, 1].legend()\n        axes[0, 1].grid(True, alpha=0.3)\n        \n        # Precision\n        if 'precision' in self.history.history:\n            axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)\n            axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)\n            axes[1, 0].set_title('Model Precision')\n            axes[1, 0].set_xlabel('Epoch')\n            axes[1, 0].set_ylabel('Precision')\n            axes[1, 0].legend()\n            axes[1, 0].grid(True, alpha=0.3)\n        \n        # Recall\n        if 'recall' in self.history.history:\n            axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)\n            axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)\n            axes[1, 1].set_title('Model Recall')\n            axes[1, 1].set_xlabel('Epoch')\n            axes[1, 1].set_ylabel('Recall')\n            axes[1, 1].legend()\n            axes[1, 1].grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.savefig(f'results/training_history_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.png', \n                   dpi=300, bbox_inches='tight')\n        plt.show()\n    \n    def plot_confusion_matrix(self, cm: np.ndarray, target_names: List[str]) -> None:\n        \"\"\"\n        Plot confusion matrix.\n        \n        Args:\n            cm (np.ndarray): Confusion matrix\n            target_names (List[str]): Class names\n        \"\"\"\n        plt.figure(figsize=(8, 6))\n        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n                   xticklabels=target_names, yticklabels=target_names)\n        plt.title('Confusion Matrix - EEG Stress Detection')\n        plt.ylabel('True Label')\n        plt.xlabel('Predicted Label')\n        plt.tight_layout()\n        plt.savefig(f'results/confusion_matrix_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.png',\n                   dpi=300, bbox_inches='tight')\n        plt.show()\n    \n    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:\n        \"\"\"\n        Make predictions on new data.\n        \n        Args:\n            X (np.ndarray): Input features\n            \n        Returns:\n            Tuple[np.ndarray, np.ndarray]: Predictions and probabilities\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not trained yet. Call train() first.\")\n        \n        # Scale features\n        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)\n        \n        # Make predictions\n        y_pred_proba = self.model.predict(X_scaled, verbose=0)\n        \n        if self.model_config['num_classes'] > 2:\n            y_pred = np.argmax(y_pred_proba, axis=1)\n        else:\n            y_pred = (y_pred_proba > 0.5).astype(int).flatten()\n        \n        return y_pred, y_pred_proba\n    \n    def save_model(self, filepath: Optional[str] = None) -> None:\n        \"\"\"\n        Save the trained model and associated parameters.\n        \n        Args:\n            filepath (Optional[str]): Custom filepath for saving\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"No model to save\")\n        \n        if filepath is None:\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            filepath = os.path.join(self.save_dir, f'cnn_lstm_stress_model_{timestamp}.h5')\n        \n        self.model.save(filepath)\n        \n        # Save additional parameters\n        import pickle\n        params_path = filepath.replace('.h5', '_params.pkl')\n        with open(params_path, 'wb') as f:\n            pickle.dump({\n                'scaler': self.scaler,\n                'label_encoder': self.label_encoder,\n                'model_config': self.model_config\n            }, f)\n        \n        logger.info(f\"Model saved to {filepath}\")\n        logger.info(f\"Parameters saved to {params_path}\")\n    \n    def load_model(self, filepath: str) -> None:\n        \"\"\"\n        Load a pre-trained model and associated parameters.\n        \n        Args:\n            filepath (str): Path to the saved model\n        \"\"\"\n        self.model = keras.models.load_model(filepath)\n        \n        # Load additional parameters\n        import pickle\n        params_path = filepath.replace('.h5', '_params.pkl')\n        \n        if os.path.exists(params_path):\n            with open(params_path, 'rb') as f:\n                params = pickle.load(f)\n                self.scaler = params['scaler']\n                self.label_encoder = params['label_encoder']\n                self.model_config = params['model_config']\n        \n        logger.info(f\"Model loaded from {filepath}\")\n    \n    def cross_validate(self, X: np.ndarray, y: np.ndarray, \n                      cv_folds: int = 5) -> List[float]:\n        \"\"\"\n        Perform cross-validation for robust model evaluation.\n        \n        Args:\n            X (np.ndarray): Input features\n            y (np.ndarray): Target labels\n            cv_folds (int): Number of cross-validation folds\n            \n        Returns:\n            List[float]: Cross-validation scores\n        \"\"\"\n        logger.info(f\"Performing {cv_folds}-fold cross-validation...\")\n        \n        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)\n        cv_scores = []\n        \n        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n            logger.info(f\"Fold {fold + 1}/{cv_folds}\")\n            \n            X_train_cv, X_val_cv = X[train_idx], X[val_idx]\n            y_train_cv, y_val_cv = y[train_idx], y[val_idx]\n            \n            # Create and train model for this fold\n            model_cv = self.build_model(X_train_cv.shape[1:])\n            \n            # Train with fewer epochs for CV\n            model_cv.fit(\n                X_train_cv, y_train_cv,\n                validation_data=(X_val_cv, y_val_cv),\n                epochs=30,\n                batch_size=16,\n                verbose=0\n            )\n            \n            # Evaluate\n            y_pred_cv = model_cv.predict(X_val_cv, verbose=0)\n            if self.model_config['num_classes'] > 2:\n                y_pred_cv = np.argmax(y_pred_cv, axis=1)\n            else:\n                y_pred_cv = (y_pred_cv > 0.5).astype(int).flatten()\n            \n            accuracy_cv = accuracy_score(y_val_cv, y_pred_cv)\n            cv_scores.append(accuracy_cv)\n            \n            logger.info(f\"Fold {fold + 1} Accuracy: {accuracy_cv:.4f}\")\n        \n        logger.info(f\"CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})\")\n        return cv_scores\n\n\nif __name__ == \"__main__\":\n    # Example usage\n    detector = CNNLSTMStressDetector()\n    \n    # Generate sample data for testing\n    n_samples, n_features = 1000, 100\n    X_sample = np.random.randn(n_samples, n_features)\n    y_sample = np.random.randint(0, 3, n_samples)\n    \n    # Train model\n    history = detector.train(X_sample, y_sample, epochs=5)\n    \n    # Evaluate model\n    results = detector.evaluate(X_sample[:200], y_sample[:200])\n    \n    print(\"Training completed successfully!\")
