import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Input, Concatenate, Attention, MultiHeadAttention,
    LayerNormalization, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedEEGStressDetector:
    def __init__(self, data_path='../complete_eeg', model_save_path='../build'):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        
        # Create build directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_and_prepare_data(self):
        """Load and prepare the EEG data with intelligent labeling"""
        try:
            # Load processed features
            features_path = os.path.join(self.data_path, 'processed_features.csv')
            features_df = pd.read_csv(features_path)
            
            # Load raw EEG data for additional analysis
            eeg_data_path = os.path.join(self.data_path, 'processed_eeg_data.csv')
            eeg_df = pd.read_csv(eeg_data_path)
            
            print(f"Loaded features shape: {features_df.shape}")
            print(f"Loaded EEG data shape: {eeg_df.shape}")
            
            # Extract features (excluding window_id)
            X = features_df.drop('window_id', axis=1).values
            
            # Create intelligent labels based on EEG characteristics
            y = self._create_stress_labels(features_df, eeg_df)
            
            print(f"Final dataset shape: X={X.shape}, y={y.shape}")
            print(f"Label distribution: {np.bincount(y)}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def _create_stress_labels(self, features_df, eeg_df):
        """Create stress labels based on EEG signal characteristics"""
        print("Creating intelligent stress labels based on EEG characteristics...")
        
        # Extract key features for stress detection
        n_samples = len(features_df)
        stress_indicators = []
        
        for i in range(n_samples):
            # Get features for this window
            window_features = features_df.iloc[i, 1:].values  # Exclude window_id
            
            # Calculate stress indicators
            # 1. High frequency power (beta/gamma bands) - stress increases these
            high_freq_power = np.mean(window_features[285:])  # Frequency domain features
            
            # 2. Signal variability - stress affects EEG variability
            signal_variability = np.std(window_features[:285])  # Time domain features
            
            # 3. Asymmetry measures - stress affects brain asymmetry
            asymmetry = np.mean(np.abs(window_features[::2] - window_features[1::2]))
            
            # 4. Signal complexity - stress changes signal complexity
            complexity = np.sum(np.abs(np.diff(window_features[:100])))
            
            # Combine indicators
            stress_score = (
                0.3 * (high_freq_power > np.percentile(features_df.iloc[:, 286:].values.flatten(), 60)) +
                0.25 * (signal_variability > np.percentile([np.std(features_df.iloc[j, 1:286].values) for j in range(n_samples)], 65)) +
                0.25 * (asymmetry > np.median([np.mean(np.abs(features_df.iloc[j, 1::2].values - features_df.iloc[j, 2::2].values)) for j in range(n_samples)])) +
                0.2 * (complexity > np.percentile([np.sum(np.abs(np.diff(features_df.iloc[j, 1:101].values))) for j in range(n_samples)], 55))
            )
            
            stress_indicators.append(stress_score)
        
        # Convert to binary labels (0: no stress, 1: stress)
        stress_threshold = np.percentile(stress_indicators, 60)  # Top 40% as stressed
        labels = (np.array(stress_indicators) > stress_threshold).astype(int)
        
        # Ensure balanced classes
        if np.sum(labels) < len(labels) * 0.3:  # If less than 30% positive
            stress_threshold = np.percentile(stress_indicators, 70)
            labels = (np.array(stress_indicators) > stress_threshold).astype(int)
        
        print(f"Stress threshold: {stress_threshold:.4f}")
        print(f"Stress labels - No stress: {np.sum(labels == 0)}, Stress: {np.sum(labels == 1)}")
        
        return labels
    
    def augment_data(self, X, y, augmentation_factor=3):
        """Advanced data augmentation for EEG signals"""
        print(f"Applying data augmentation (factor: {augmentation_factor})...")
        
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(augmentation_factor):
            X_temp = X.copy()
            
            # 1. Gaussian noise injection
            noise_std = 0.05 * np.std(X, axis=0)
            X_temp += np.random.normal(0, noise_std, X_temp.shape)
            
            # 2. Feature scaling augmentation
            scale_factors = np.random.uniform(0.9, 1.1, X_temp.shape[1])
            X_temp *= scale_factors
            
            # 3. Feature dropout (randomly set some features to zero)
            dropout_mask = np.random.random(X_temp.shape) > 0.05
            X_temp *= dropout_mask
            
            # 4. Frequency domain augmentation (for frequency features)
            freq_features = X_temp[:, 285:]
            freq_noise = np.random.normal(0, 0.02 * np.std(freq_features, axis=0), freq_features.shape)
            X_temp[:, 285:] += freq_noise
            
            X_aug.append(X_temp)
            y_aug.append(y)
        
        X_augmented = np.vstack(X_aug)
        y_augmented = np.hstack(y_aug)
        
        print(f"Augmented dataset shape: X={X_augmented.shape}, y={y_augmented.shape}")
        return X_augmented, y_augmented
    
    def preprocess_data(self, X, y, test_size=0.2, augment=True):
        """Preprocess the data with advanced techniques"""
        print("Preprocessing data...")
        
        # Split the data FIRST
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply augmentation ONLY to training data
        if augment:
            X_train, y_train = self.augment_data(X_train, y_train)
            print(f"Training data after augmentation: {X_train.shape}")
            print(f"Test data (no augmentation): {X_test.shape}")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Reshape for CNN-LSTM (samples, timesteps, features)
        n_timesteps = 10
        n_features_per_step = X_train_scaled.shape[1] // n_timesteps
        
        # Pad if necessary
        if X_train_scaled.shape[1] % n_timesteps != 0:
            pad_size = n_timesteps - (X_train_scaled.shape[1] % n_timesteps)
            X_train_scaled = np.pad(X_train_scaled, ((0, 0), (0, pad_size)), mode='constant')
            X_test_scaled = np.pad(X_test_scaled, ((0, 0), (0, pad_size)), mode='constant')
            n_features_per_step = X_train_scaled.shape[1] // n_timesteps
        
        X_train_reshaped = X_train_scaled.reshape(-1, n_timesteps, n_features_per_step)
        X_test_reshaped = X_test_scaled.reshape(-1, n_timesteps, n_features_per_step)
        
        print(f"Final preprocessed shapes - Train: {X_train_reshaped.shape}, Test: {X_test_reshaped.shape}")
        
        return X_train_reshaped, X_test_reshaped, y_train, y_test
    
    def create_advanced_model(self, input_shape):
        """Create an advanced CNN-LSTM-Transformer hybrid model"""
        print(f"Creating advanced model with input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # CNN branch for local feature extraction
        cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Dropout(0.3)(cnn_branch)
        
        cnn_branch = Conv1D(256, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = Dropout(0.3)(cnn_branch)
        
        # LSTM branch for temporal dependencies
        lstm_branch = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_branch, lstm_branch)
        attention = LayerNormalization()(attention + lstm_branch)
        attention = Dropout(0.3)(attention)
        
        # Combine CNN and LSTM-Attention branches
        # Ensure compatible shapes
        cnn_pooled = GlobalAveragePooling1D()(cnn_branch)
        lstm_pooled = GlobalAveragePooling1D()(attention)
        
        # Concatenate features
        combined = Concatenate()([cnn_pooled, lstm_pooled])
        
        # Dense layers for classification
        dense = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
        
        dense = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(dense)
        dense = Dropout(0.3)(dense)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test, epochs=100):
        """Train the model with advanced callbacks"""
        print("Training the model...")
        
        # Create the model
        self.model = self.create_advanced_model(X_train.shape[1:])
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_eeg_stress_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=16,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        final_model_path = os.path.join(self.model_save_path, 'eeg_stress_detector_final.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\nEvaluating the model...")
        
        if self.model is None:
            print("No model found. Please train the model first.")
            return
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return accuracy, f1, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Stress', 'Stress'],
                   yticklabels=['No Stress', 'Stress'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.model_save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{cv_folds}")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Scale data
            scaler_cv = RobustScaler()
            X_train_cv = scaler_cv.fit_transform(X_train_cv)
            X_val_cv = scaler_cv.transform(X_val_cv)
            
            # Reshape for model
            n_timesteps = 10
            n_features_per_step = X_train_cv.shape[1] // n_timesteps
            
            if X_train_cv.shape[1] % n_timesteps != 0:
                pad_size = n_timesteps - (X_train_cv.shape[1] % n_timesteps)
                X_train_cv = np.pad(X_train_cv, ((0, 0), (0, pad_size)), mode='constant')
                X_val_cv = np.pad(X_val_cv, ((0, 0), (0, pad_size)), mode='constant')
                n_features_per_step = X_train_cv.shape[1] // n_timesteps
            
            X_train_cv = X_train_cv.reshape(-1, n_timesteps, n_features_per_step)
            X_val_cv = X_val_cv.reshape(-1, n_timesteps, n_features_per_step)
            
            # Create and train model
            model_cv = self.create_advanced_model(X_train_cv.shape[1:])
            
            # Train with fewer epochs for CV
            model_cv.fit(
                X_train_cv, y_train_cv,
                validation_data=(X_val_cv, y_val_cv),
                epochs=30,
                batch_size=16,
                verbose=0
            )
            
            # Evaluate
            y_pred_cv = (model_cv.predict(X_val_cv) > 0.5).astype(int).flatten()
            accuracy_cv = accuracy_score(y_val_cv, y_pred_cv)
            cv_scores.append(accuracy_cv)
            
            print(f"Fold {fold + 1} Accuracy: {accuracy_cv:.4f}")
        
        print(f"\nCross-validation results:")
        print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        return cv_scores

def main():
    """Main execution function"""
    print("=== Advanced EEG Stress Detection Model ===")
    print("Loading and processing complete_eeg dataset...\n")
    
    try:
        # Initialize the detector with correct paths
        detector = AdvancedEEGStressDetector(
            data_path='../complete_eeg',
            model_save_path='../build'
        )
        
        # Load and prepare data
        X, y = detector.load_and_prepare_data()
        
        if X is None or y is None:
            print("Failed to load data. Exiting.")
            return
        
        # Preprocess data
        # In the main function, change:
        X_train, X_test, y_train, y_test = detector.preprocess_data(X, y, test_size=0.3, augment=True)
        
        # Train the model
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        history = detector.train_model(X_train, X_test, y_train, y_test, epochs=100)
        
        # Evaluate the model
        print("\n" + "="*50)
        print("EVALUATION PHASE")
        print("="*50)
        
        accuracy, f1, y_pred, y_pred_proba = detector.evaluate_model(X_test, y_test)
        
        # Plot training history
        detector.plot_training_history()
        
        # Perform cross-validation on original data (without augmentation)
        print("\n" + "="*50)
        print("CROSS-VALIDATION PHASE")
        print("="*50)
        
        cv_scores = detector.cross_validate(X, y)
        
        # Final summary
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Cross-validation Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        print(f"\nModel saved in H5 format at: build/eeg_stress_detector_final.h5")
        print(f"Best model checkpoint: build/best_eeg_stress_model.h5")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()