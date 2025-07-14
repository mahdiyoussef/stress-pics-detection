import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedEEGStressDetector:
    def __init__(self, data_dir='eeg_mental_state'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # Create build directory if it doesn't exist
        self.build_dir = '../build'
        os.makedirs(self.build_dir, exist_ok=True)
        
    def advanced_data_augmentation(self, X, y, augment_factor=10):
        """Advanced EEG data augmentation techniques based on research"""
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            signal = X[i]
            label = y[i]
            
            # Original signal
            augmented_X.append(signal)
            augmented_y.append(label)
            
            for _ in range(augment_factor):
                aug_signal = signal.copy()
                
                # 1. Gaussian Noise Addition (proven effective)
                noise_factor = np.random.uniform(0.01, 0.05)
                aug_signal += np.random.normal(0, noise_factor, aug_signal.shape)
                
                # 2. Time Shifting (circular shift)
                shift = np.random.randint(-len(signal)//4, len(signal)//4)
                aug_signal = np.roll(aug_signal, shift, axis=0)
                
                # 3. Amplitude Scaling
                scale_factor = np.random.uniform(0.8, 1.2)
                aug_signal *= scale_factor
                
                # 4. Frequency Domain Augmentation
                if np.random.random() > 0.5:
                    fft_signal = np.fft.fft(aug_signal, axis=0)
                    # Add noise in frequency domain
                    freq_noise = np.random.normal(0, 0.01, fft_signal.shape)
                    fft_signal += freq_noise
                    aug_signal = np.real(np.fft.ifft(fft_signal, axis=0))
                
                # 5. Channel Dropout (for multi-channel EEG)
                if len(aug_signal.shape) > 1 and np.random.random() > 0.7:
                    dropout_channels = np.random.choice(aug_signal.shape[1], 
                                                      size=max(1, aug_signal.shape[1]//4), 
                                                      replace=False)
                    aug_signal[:, dropout_channels] = 0
                
                augmented_X.append(aug_signal)
                augmented_y.append(label)
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def create_robust_stress_labels(self, eeg_data):
        """Create stress labels based on EEG signal characteristics"""
        labels = []
        
        for signal in eeg_data:
            # Flatten if multi-dimensional
            if len(signal.shape) > 1:
                flat_signal = signal.flatten()
            else:
                flat_signal = signal
            
            # Calculate multiple stress indicators
            variance = np.var(flat_signal)
            mean_abs = np.mean(np.abs(flat_signal))
            energy = np.sum(flat_signal ** 2)
            
            # Frequency domain features
            fft_signal = np.fft.fft(flat_signal)
            power_spectrum = np.abs(fft_signal) ** 2
            
            # Beta/Alpha ratio (stress indicator)
            freqs = np.fft.fftfreq(len(flat_signal), 1/250)  # Assuming 250Hz sampling
            alpha_power = np.mean(power_spectrum[(freqs >= 8) & (freqs <= 12)])
            beta_power = np.mean(power_spectrum[(freqs >= 13) & (freqs <= 30)])
            
            beta_alpha_ratio = beta_power / (alpha_power + 1e-8)
            
            # Combine features for stress classification
            stress_score = (variance * 0.3 + mean_abs * 0.2 + 
                          energy * 0.2 + beta_alpha_ratio * 0.3)
            
            # Create 3 stress levels
            if stress_score < np.percentile([stress_score], 33):
                labels.append(0)  # Low stress
            elif stress_score < np.percentile([stress_score], 66):
                labels.append(1)  # Medium stress
            else:
                labels.append(2)  # High stress
        
        return np.array(labels)
    
    def load_and_prepare_data(self):
        """Load and prepare EEG data with comprehensive preprocessing"""
        try:
            # Try to load real EEG data
            eeg_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
            
            if not eeg_files:
                print(f"No CSV files found in {self.data_dir}. Generating comprehensive synthetic data...")
                return self.generate_comprehensive_synthetic_data()
            
            all_data = []
            for file in eeg_files:
                try:
                    data = pd.read_csv(file)
                    if not data.empty:
                        all_data.append(data.values)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
            
            if not all_data:
                print("No valid data loaded. Generating synthetic data...")
                return self.generate_comprehensive_synthetic_data()
            
            # Combine all data
            combined_data = np.vstack(all_data)
            print(f"Loaded real EEG data: {combined_data.shape}")
            
            # Ensure minimum samples
            if len(combined_data) < 100:
                print("Insufficient real data. Augmenting with synthetic data...")
                synthetic_data = self.generate_comprehensive_synthetic_data(n_samples=500)
                combined_data = np.vstack([combined_data, synthetic_data[0]])
            
            # Create labels
            labels = self.create_robust_stress_labels([combined_data[i:i+1000] 
                                                     for i in range(0, len(combined_data)-1000, 100)])
            
            # Prepare sequences
            X = []
            y = []
            sequence_length = 1000
            
            for i in range(0, len(combined_data) - sequence_length, 50):
                X.append(combined_data[i:i+sequence_length])
                y.append(labels[min(i//100, len(labels)-1)])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error in data loading: {e}")
            return self.generate_comprehensive_synthetic_data()
    
    def generate_comprehensive_synthetic_data(self, n_samples=1000):
        """Generate comprehensive synthetic EEG data with realistic characteristics"""
        print(f"Generating {n_samples} synthetic EEG samples...")
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Generate realistic EEG signal
            t = np.linspace(0, 4, 1000)  # 4 seconds at 250Hz
            
            # Base EEG components
            alpha = np.sin(2 * np.pi * 10 * t) * np.random.uniform(0.5, 1.5)
            beta = np.sin(2 * np.pi * 20 * t) * np.random.uniform(0.3, 1.0)
            theta = np.sin(2 * np.pi * 6 * t) * np.random.uniform(0.2, 0.8)
            
            # Stress-related modifications
            stress_level = i % 3
            
            if stress_level == 0:  # Low stress
                signal = alpha * 1.2 + theta * 0.8 + beta * 0.5
                noise_level = 0.1
            elif stress_level == 1:  # Medium stress
                signal = alpha * 1.0 + theta * 0.6 + beta * 0.8
                noise_level = 0.15
            else:  # High stress
                signal = alpha * 0.8 + theta * 0.4 + beta * 1.2
                noise_level = 0.2
            
            # Add realistic noise
            noise = np.random.normal(0, noise_level, len(signal))
            signal += noise
            
            # Add artifacts occasionally
            if np.random.random() > 0.8:
                artifact_pos = np.random.randint(0, len(signal)-50)
                signal[artifact_pos:artifact_pos+50] += np.random.uniform(-2, 2)
            
            X.append(signal.reshape(-1, 1))
            y.append(stress_level)
        
        return np.array(X), np.array(y)
    
    def create_advanced_cnn_lstm_model(self, input_shape, num_classes=3):
        """Create advanced CNN-LSTM model with research-based architecture"""
        model = models.Sequential([
            # Multi-scale CNN feature extraction
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(32, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # LSTM for temporal dependencies
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            
            # Dense layers with strong regularization
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Advanced optimizer with learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_advanced_techniques(self, X, y, validation_split=0.2, epochs=200):
        """Train model with advanced techniques to prevent overfitting"""
        # Apply data augmentation
        print("Applying advanced data augmentation...")
        X_aug, y_aug = self.advanced_data_augmentation(X, y, augment_factor=5)
        
        # Normalize data
        X_aug_reshaped = X_aug.reshape(X_aug.shape[0], -1)
        X_aug_normalized = self.scaler.fit_transform(X_aug_reshaped)
        X_aug_final = X_aug_normalized.reshape(X_aug.shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug_final, y_aug, test_size=validation_split, 
            stratify=y_aug, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Create model
        self.model = self.create_advanced_cnn_lstm_model(X_train.shape[1:])
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=30,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,  # Increased patience
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                os.path.join(self.build_dir, 'best_eeg_mental_state_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("Starting training with advanced techniques...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=min(32, len(X_train)//4),
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save the final model
        final_model_path = os.path.join(self.build_dir, 'eeg_mental_state_final_model.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return X_val, y_val
    
    def evaluate_comprehensive(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if self.model is None:
            print("No model to evaluate!")
            return
        
        try:
            # Predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"\n=== COMPREHENSIVE EVALUATION RESULTS ===")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test samples: {len(y_test)}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Low Stress', 'Medium Stress', 'High Stress']))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Low', 'Medium', 'High'],
                       yticklabels=['Low', 'Medium', 'High'])
            plt.title('Confusion Matrix - Advanced EEG Stress Detection')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            confusion_matrix_path = os.path.join(self.build_dir, f'eeg_mental_state_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
            plt.close()
            
            return accuracy, f1
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return 0.0, 0.0
    
    def plot_advanced_training_history(self):
        """Plot comprehensive training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'learning_rate' in self.history.history:
            axes[1, 0].plot(self.history.history['learning_rate'], linewidth=2, color='red')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training Progress
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        axes[1, 1].plot(epochs, self.history.history['accuracy'], 'b-', label='Training', linewidth=2)
        axes[1, 1].plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[1, 1].fill_between(epochs, self.history.history['accuracy'], alpha=0.3)
        axes[1, 1].fill_between(epochs, self.history.history['val_accuracy'], alpha=0.3)
        axes[1, 1].set_title('Training Progress')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../build/advanced_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        plt.close()

def main():
    """Main execution function with comprehensive error handling"""
    try:
        print("=== ADVANCED EEG STRESS DETECTION SYSTEM ===")
        print("Implementing research-based solutions for small datasets\n")
        
        # Initialize detector
        detector = AdvancedEEGStressDetector()
        
        # Load and prepare data
        print("Loading and preparing data...")
        X, y = detector.load_and_prepare_data()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Train model with advanced techniques
        print("\nTraining model with advanced techniques...")
        X_val, y_val = detector.train_with_advanced_techniques(X, y, epochs=200)
        
        # Evaluate model
        print("\nEvaluating model...")
        accuracy, f1_score = detector.evaluate_comprehensive(X_val, y_val)
        
        # Plot training history
        detector.plot_advanced_training_history()
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Advanced CNN-LSTM Model:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1-Score: {f1_score:.4f}")
        print(f"  - Training completed with {len(detector.history.history['accuracy'])} epochs")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()