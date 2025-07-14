import numpy as np
import pandas as pd
import os
import sys
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import load_datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_datasets import load_eeg_as_dataframe

class EEGMentalStatePreprocessor:
    """
    Robust preprocessing pipeline for EEG Mental State dataset
    Specifically designed for stress detection and mental state classification
    """
    
    def __init__(self, sampling_rate=256, target_rate=128):
        self.sampling_rate = sampling_rate
        self.target_rate = target_rate
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.ica = None
        self.preprocessing_params = {}
        
    def load_dataset(self):
        """
        Load the EEG mental state dataset
        """
        try:
            print("Loading EEG Mental State dataset...")
            dataset_handle = 'birdy654/eeg-brainwave-dataset-mental-state'
            
            # Try to load as dataframe first
            data_path = load_eeg_as_dataframe(dataset_handle)
            
            if isinstance(data_path, str):
                # If path returned, look for CSV files
                csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
                    print(f"Loaded dataset with shape: {df.shape}")
                    return df
                else:
                    print(f"Dataset downloaded to: {data_path}")
                    print("Please check the dataset structure manually.")
                    return None
            else:
                return data_path
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def detect_channels(self, df):
        """
        Automatically detect EEG channels in the dataset
        """
        # Common EEG channel names
        eeg_patterns = ['EEG', 'eeg', 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 
                       'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                       'Fz', 'Cz', 'Pz', 'AF3', 'AF4', 'FC1', 'FC2', 'CP1', 'CP2']
        
        eeg_channels = []
        for col in df.columns:
            if any(pattern in col for pattern in eeg_patterns):
                eeg_channels.append(col)
        
        if not eeg_channels:
            # If no standard names found, assume numeric columns are EEG
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove obvious non-EEG columns
            exclude_patterns = ['time', 'timestamp', 'label', 'target', 'class', 'id']
            eeg_channels = [col for col in numeric_cols 
                          if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        print(f"Detected {len(eeg_channels)} EEG channels: {eeg_channels[:10]}...")
        return eeg_channels
    
    def apply_bandpass_filter(self, data, low_freq=0.5, high_freq=50):
        """
        Apply bandpass filter to remove artifacts and noise
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def apply_notch_filter(self, data, notch_freq=50, quality_factor=30):
        """
        Apply notch filter to remove power line interference
        """
        # Design notch filter
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def remove_artifacts_ica(self, data, n_components=None):
        """
        Remove artifacts using Independent Component Analysis (ICA)
        """
        if n_components is None:
            n_components = min(data.shape[1], 15)
        
        # Apply ICA
        self.ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        
        # Fit and transform
        ica_components = self.ica.fit_transform(data.T).T
        
        # Automatically detect artifact components (simplified approach)
        # In practice, you might want more sophisticated artifact detection
        artifact_threshold = 3  # Standard deviations
        clean_components = []
        
        for i, component in enumerate(ica_components.T):
            if np.abs(zscore(component)).max() < artifact_threshold:
                clean_components.append(i)
        
        # Reconstruct data using only clean components
        if clean_components:
            clean_ica = np.zeros_like(ica_components)
            clean_ica[:, clean_components] = ica_components[:, clean_components]
            cleaned_data = self.ica.inverse_transform(clean_ica.T).T
        else:
            cleaned_data = data
        
        print(f"ICA: Removed {n_components - len(clean_components)} artifact components")
        return cleaned_data
    
    def downsample_data(self, data):
        """
        Downsample data to target sampling rate
        """
        if self.target_rate >= self.sampling_rate:
            return data
        
        downsample_factor = self.sampling_rate // self.target_rate
        downsampled_data = data[::downsample_factor, :]
        
        print(f"Downsampled from {self.sampling_rate}Hz to {self.target_rate}Hz")
        return downsampled_data
    
    def extract_time_domain_features(self, data, window_size=512, overlap=256):
        """
        Extract time-domain features from EEG data
        """
        features = []
        n_samples, n_channels = data.shape
        
        # Sliding window approach
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Statistical features
                window_features.extend([
                    np.mean(channel_data),                    # Mean
                    np.std(channel_data),                     # Standard deviation
                    np.var(channel_data),                     # Variance
                    np.max(channel_data) - np.min(channel_data),  # Peak-to-peak
                    np.percentile(channel_data, 75) - np.percentile(channel_data, 25),  # IQR
                    np.mean(np.abs(channel_data)),            # Mean absolute value
                    np.sqrt(np.mean(channel_data**2)),        # RMS
                    len(np.where(np.diff(np.sign(channel_data)))[0]),  # Zero crossings
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, data, window_size=512, overlap=256):
        """
        Extract frequency-domain features from EEG data
        """
        features = []
        n_samples, n_channels = data.shape
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Sliding window approach
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Power Spectral Density
                freqs, psd = signal.welch(channel_data, self.target_rate, nperseg=min(256, len(channel_data)))
                
                # Band power features
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        window_features.append(band_power)
                    else:
                        window_features.append(0)
                
                # Spectral features
                total_power = np.trapz(psd, freqs)
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
                
                window_features.extend([total_power, spectral_centroid, spectral_bandwidth])
            
            features.append(window_features)
        
        return np.array(features)
    
    def normalize_features(self, features, method='standard'):
        """
        Normalize extracted features
        """
        if method == 'standard':
            normalized_features = self.scaler.fit_transform(features)
        elif method == 'robust':
            normalized_features = self.robust_scaler.fit_transform(features)
        else:
            # Min-max normalization
            normalized_features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
        
        return normalized_features
    
    def preprocess_pipeline(self, df=None, save_results=True):
        """
        Complete preprocessing pipeline for EEG mental state dataset
        """
        print("Starting EEG Mental State preprocessing pipeline...")
        
        # Load dataset if not provided
        if df is None:
            df = self.load_dataset()
            if df is None:
                return None
        
        # Detect EEG channels
        eeg_channels = self.detect_channels(df)
        if not eeg_channels:
            print("No EEG channels detected!")
            return None
        
        # Extract EEG data
        eeg_data = df[eeg_channels].values
        print(f"EEG data shape: {eeg_data.shape}")
        
        # Step 1: Apply bandpass filter
        print("Applying bandpass filter (0.5-50 Hz)...")
        filtered_data = self.apply_bandpass_filter(eeg_data)
        
        # Step 2: Apply notch filter
        print("Applying notch filter (50 Hz)...")
        filtered_data = self.apply_notch_filter(filtered_data)
        
        # Step 3: Remove artifacts using ICA
        print("Removing artifacts using ICA...")
        cleaned_data = self.remove_artifacts_ica(filtered_data)
        
        # Step 4: Downsample data
        print("Downsampling data...")
        downsampled_data = self.downsample_data(cleaned_data)
        
        # Step 5: Extract features
        print("Extracting time-domain features...")
        time_features = self.extract_time_domain_features(downsampled_data)
        
        print("Extracting frequency-domain features...")
        freq_features = self.extract_frequency_domain_features(downsampled_data)
        
        # Combine features
        all_features = np.hstack([time_features, freq_features])
        print(f"Combined features shape: {all_features.shape}")
        
        # Step 6: Normalize features
        print("Normalizing features...")
        normalized_features = self.normalize_features(all_features)
        
        # Save preprocessing parameters
        self.preprocessing_params = {
            'eeg_channels': eeg_channels,
            'original_shape': eeg_data.shape,
            'processed_shape': normalized_features.shape,
            'sampling_rate': self.sampling_rate,
            'target_rate': self.target_rate,
            'n_features': normalized_features.shape[1]
        }
        
        if save_results:
            self.save_processed_data(normalized_features, downsampled_data)
        
        print("Preprocessing completed successfully!")
        return {
            'features': normalized_features,
            'processed_eeg': downsampled_data,
            'params': self.preprocessing_params
        }
    
    def save_processed_data(self, features, processed_eeg):
        """
        Save processed data and parameters as CSV files
        """
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Convert features to DataFrame and save as CSV
        print("Saving processed features as CSV...")
        
        # Create feature column names
        n_channels = len(self.preprocessing_params.get('eeg_channels', []))
        feature_names = []
        
        # Time-domain feature names
        time_features = ['mean', 'std', 'var', 'peak_to_peak', 'iqr', 'mav', 'rms', 'zero_crossings']
        for ch_idx in range(n_channels):
            for feat in time_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Frequency-domain feature names
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        freq_features = bands + ['total_power', 'spectral_centroid', 'spectral_bandwidth']
        for ch_idx in range(n_channels):
            for feat in freq_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Ensure we have the right number of feature names
        if len(feature_names) != features.shape[1]:
            # Fallback to generic names if mismatch
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        # Create DataFrame with features
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Add window index
        features_df.insert(0, 'window_id', range(len(features_df)))
        
        # Save features as CSV
        features_csv_path = os.path.join(output_dir, 'processed_features.csv')
        features_df.to_csv(features_csv_path, index=False)
        print(f"Features saved to: {features_csv_path}")
        
        # Save processed EEG data as CSV
        print("Saving processed EEG data as CSV...")
        
        # Create channel names for EEG data
        eeg_channels = self.preprocessing_params.get('eeg_channels', [])
        if len(eeg_channels) != processed_eeg.shape[1]:
            eeg_channels = [f'channel_{i}' for i in range(processed_eeg.shape[1])]
        
        # Create DataFrame with processed EEG
        eeg_df = pd.DataFrame(processed_eeg, columns=eeg_channels)
        
        # Add time index (assuming target sampling rate)
        time_points = np.arange(len(eeg_df)) / self.target_rate
        eeg_df.insert(0, 'time_seconds', time_points)
        
        # Save EEG data as CSV
        eeg_csv_path = os.path.join(output_dir, 'processed_eeg_data.csv')
        eeg_df.to_csv(eeg_csv_path, index=False)
        print(f"Processed EEG data saved to: {eeg_csv_path}")
        
        # Also save as numpy arrays (backup)
        np.save(os.path.join(output_dir, 'processed_features.npy'), features)
        np.save(os.path.join(output_dir, 'processed_eeg_data.npy'), processed_eeg)
        
        # Save preprocessing parameters as JSON
        import json
        with open(os.path.join(output_dir, 'preprocessing_params.json'), 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            params_serializable = {}
            for key, value in self.preprocessing_params.items():
                if isinstance(value, np.ndarray):
                    params_serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    params_serializable[key] = value.item()
                else:
                    params_serializable[key] = value
            json.dump(params_serializable, f, indent=2)
        
        # Create a summary CSV with dataset information
        summary_data = {
            'dataset_name': ['EEG Mental State'],
            'original_samples': [self.preprocessing_params.get('original_shape', [0, 0])[0]],
            'original_channels': [self.preprocessing_params.get('original_shape', [0, 0])[1]],
            'processed_windows': [features.shape[0]],
            'features_per_window': [features.shape[1]],
            'sampling_rate_hz': [self.target_rate],
            'preprocessing_date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'dataset_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Dataset summary saved to: {summary_csv_path}")
        
        print(f"\nAll processed data saved to: {output_dir}")
        print("Files created:")
        print(f"  - processed_features.csv ({features.shape[0]} windows × {features.shape[1]} features)")
        print(f"  - processed_eeg_data.csv ({processed_eeg.shape[0]} samples × {processed_eeg.shape[1]} channels)")
        print(f"  - dataset_summary.csv (metadata)")
        print(f"  - preprocessing_params.json (parameters)")
    
    def load_processed_data_csv(self):
        """
        Load previously processed data from CSV files
        """
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Load features from CSV
            features_csv_path = os.path.join(output_dir, 'processed_features.csv')
            features_df = pd.read_csv(features_csv_path)
            
            # Remove window_id column if present
            if 'window_id' in features_df.columns:
                features = features_df.drop('window_id', axis=1).values
            else:
                features = features_df.values
            
            # Load processed EEG from CSV
            eeg_csv_path = os.path.join(output_dir, 'processed_eeg_data.csv')
            eeg_df = pd.read_csv(eeg_csv_path)
            
            # Remove time column if present
            if 'time_seconds' in eeg_df.columns:
                processed_eeg = eeg_df.drop('time_seconds', axis=1).values
            else:
                processed_eeg = eeg_df.values
            
            # Load parameters
            import json
            with open(os.path.join(output_dir, 'preprocessing_params.json'), 'r') as f:
                params = json.load(f)
            
            print(f"Loaded processed data from CSV files:")
            print(f"  - Features: {features.shape}")
            print(f"  - EEG data: {processed_eeg.shape}")
            
            return {
                'features': features,
                'processed_eeg': processed_eeg,
                'params': params,
                'features_df': features_df,
                'eeg_df': eeg_df
            }
            
        except FileNotFoundError as e:
            print(f"CSV files not found: {e}")
            print("Run preprocessing first or check file paths.")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = EEGMentalStatePreprocessor(sampling_rate=256, target_rate=128)
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline()
    
    if results:
        print("\n=== Preprocessing Results ===")
        print(f"Features shape: {results['features'].shape}")
        print(f"Processed EEG shape: {results['processed_eeg'].shape}")
        print(f"Number of features per window: {results['params']['n_features']}")
        
        # Basic statistics
        features = results['features']
        print(f"\nFeature statistics:")
        print(f"Mean: {np.mean(features):.4f}")
        print(f"Std: {np.std(features):.4f}")
        print(f"Min: {np.min(features):.4f}")
        print(f"Max: {np.max(features):.4f}")
        
        # Check for NaN values
        nan_count = np.isnan(features).sum()
        print(f"NaN values: {nan_count}")
        
        if nan_count == 0:
            print("\n✅ Preprocessing completed successfully!")
            print("Data saved as CSV files and ready for model training.")
            
            # Test loading CSV data
            print("\n=== Testing CSV Loading ===")
            loaded_data = preprocessor.load_processed_data_csv()
            if loaded_data:
                print("✅ CSV data loaded successfully!")
        else:
            print("\n⚠️ Warning: NaN values detected in features.")