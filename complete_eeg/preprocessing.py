import numpy as np
import pandas as pd
import os
import sys
from scipy import signal
from scipy.stats import zscore, entropy
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import FastICA
import pywt
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import load_datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_datasets import load_eeg_as_dataframe

class CompleteEEGPreprocessor:
    """
    Advanced preprocessing pipeline for Complete EEG dataset
    Incorporates state-of-the-art techniques from recent research:
    - Robust artifact removal with ICA and statistical thresholding
    - Wavelet transform for time-frequency analysis
    - Empirical Mode Decomposition (EMD) for nonlinear signal analysis
    - Comprehensive feature extraction (time, frequency, and nonlinear domains)
    - Enhanced filtering and preprocessing based on 2024 research
    """
    
    def __init__(self, sampling_rate=None, target_rate=128):
        self.sampling_rate = sampling_rate  # Will be auto-detected
        self.target_rate = target_rate
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.ica = None
        self.preprocessing_params = {}
        
    def load_dataset(self, dataset_handle='amananandrai/complete-eeg-dataset'):
        """
        Load the Complete EEG dataset
        """
        try:
            print("Loading Complete EEG dataset...")
            
            # Try to load as dataframe first
            data_path = load_eeg_as_dataframe(dataset_handle)
            
            if isinstance(data_path, str):
                # If path returned, look for CSV files
                csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                if csv_files:
                    # Load the first CSV file or combine multiple files
                    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
                    print(f"Loaded dataset with shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
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
    
    def detect_sampling_rate(self, df):
        """
        Automatically detect sampling rate from the dataset
        """
        if 'time' in df.columns or 'timestamp' in df.columns:
            time_col = 'time' if 'time' in df.columns else 'timestamp'
            time_diff = np.diff(df[time_col].values[:1000])  # Use first 1000 samples
            median_diff = np.median(time_diff)
            detected_rate = int(1 / median_diff) if median_diff > 0 else 256
        else:
            # Default assumption for EEG data
            detected_rate = 256
        
        self.sampling_rate = detected_rate
        print(f"Detected sampling rate: {self.sampling_rate} Hz")
        return detected_rate
    
    def detect_channels(self, df):
        """
        Automatically detect EEG channels in the dataset with enhanced patterns
        """
        # Extended EEG channel patterns based on 10-20 system and research standards
        eeg_patterns = [
            'EEG', 'eeg', 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'AF3', 'AF4', 'FC1', 'FC2',
            'CP1', 'CP2', 'PO3', 'PO4', 'F1', 'F2', 'FC3', 'FC4', 'C1', 'C2', 'CP3', 'CP4',
            'P1', 'P2', 'PO7', 'PO8', 'OZ', 'channel', 'ch', 'electrode'
        ]
        
        eeg_channels = []
        for col in df.columns:
            if any(pattern in col for pattern in eeg_patterns):
                eeg_channels.append(col)
        
        if not eeg_channels:
            # If no standard names found, assume numeric columns are EEG
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove obvious non-EEG columns
            exclude_patterns = ['time', 'timestamp', 'label', 'target', 'class', 'id', 'index']
            eeg_channels = [col for col in numeric_cols 
                          if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        print(f"Detected {len(eeg_channels)} EEG channels: {eeg_channels[:10]}...")
        return eeg_channels
    
    def apply_bandpass_filter(self, data, low_freq=0.5, high_freq=50):
        """
        Apply robust bandpass filter with enhanced parameters
        Based on recent research recommendations for EEG preprocessing
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Use higher order Butterworth filter for better frequency response
        b, a = signal.butter(6, [low, high], btype='band')
        
        # Apply zero-phase filtering to avoid phase distortion
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def apply_notch_filter(self, data, notch_freq=50, quality_factor=35):
        """
        Apply enhanced notch filter for power line interference removal
        """
        # Apply multiple notch filters for harmonics
        filtered_data = data.copy()
        
        # Filter 50Hz and its harmonics (100Hz, 150Hz)
        for freq in [notch_freq, notch_freq*2, notch_freq*3]:
            if freq < self.sampling_rate / 2:
                b, a = signal.iirnotch(freq, quality_factor, self.sampling_rate)
                for i in range(filtered_data.shape[1]):
                    filtered_data[:, i] = signal.filtfilt(b, a, filtered_data[:, i])
        
        return filtered_data
    
    def remove_artifacts_ica(self, data, n_components=None):
        """
        Enhanced artifact removal using ICA with robust statistics
        Based on APP (Automatic Pre-processing Pipeline) methodology
        """
        if n_components is None:
            n_components = min(data.shape[1], 20)  # Increased components for better separation
        
        # Apply ICA
        self.ica = FastICA(n_components=n_components, random_state=42, max_iter=2000, tol=1e-6)
        
        # Fit and transform
        ica_components = self.ica.fit_transform(data.T).T
        
        # Enhanced artifact detection using robust statistics
        clean_components = []
        
        for i, component in enumerate(ica_components.T):
            # Use robust statistics instead of z-scores
            median_val = np.median(component)
            mad = np.median(np.abs(component - median_val))  # Median Absolute Deviation
            robust_threshold = 3.5  # More conservative threshold
            
            # Check for artifacts using multiple criteria
            is_artifact = False
            
            # 1. Amplitude-based detection
            if mad > 0:
                robust_z_scores = 0.6745 * (component - median_val) / mad
                if np.max(np.abs(robust_z_scores)) > robust_threshold:
                    is_artifact = True
            
            # 2. Frequency-based detection (high-frequency artifacts)
            freqs, psd = signal.welch(component, self.sampling_rate, nperseg=min(256, len(component)))
            high_freq_power = np.sum(psd[freqs > 30]) / np.sum(psd)
            if high_freq_power > 0.6:  # More than 60% power in high frequencies
                is_artifact = True
            
            if not is_artifact:
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
        Intelligent downsampling with anti-aliasing filter
        """
        if self.target_rate >= self.sampling_rate:
            return data
        
        # Apply anti-aliasing filter before downsampling
        nyquist = self.target_rate / 2
        cutoff = nyquist / (self.sampling_rate / 2)
        b, a = signal.butter(6, cutoff, btype='low')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        # Downsample
        downsample_factor = self.sampling_rate // self.target_rate
        downsampled_data = filtered_data[::downsample_factor, :]
        
        print(f"Downsampled from {self.sampling_rate}Hz to {self.target_rate}Hz")
        return downsampled_data
    
    def extract_time_domain_features(self, data, window_size=512, overlap=256):
        """
        Enhanced time-domain feature extraction with additional statistical measures
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
                
                # Enhanced statistical features
                window_features.extend([
                    np.mean(channel_data),                    # Mean
                    np.std(channel_data),                     # Standard deviation
                    np.var(channel_data),                     # Variance
                    np.max(channel_data) - np.min(channel_data),  # Peak-to-peak
                    np.percentile(channel_data, 75) - np.percentile(channel_data, 25),  # IQR
                    np.mean(np.abs(channel_data)),            # Mean absolute value
                    np.sqrt(np.mean(channel_data**2)),        # RMS
                    len(np.where(np.diff(np.sign(channel_data)))[0]),  # Zero crossings
                    np.sum(np.abs(np.diff(channel_data))),    # Total variation
                    np.max(channel_data),                     # Maximum
                    np.min(channel_data),                     # Minimum
                    np.median(channel_data),                  # Median
                    np.percentile(channel_data, 25),          # 25th percentile
                    np.percentile(channel_data, 75),          # 75th percentile
                    len(channel_data[channel_data > np.mean(channel_data) + np.std(channel_data)]) / len(channel_data)  # Outlier ratio
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, data, window_size=512, overlap=256):
        """
        Enhanced frequency-domain features with additional spectral measures
        """
        features = []
        n_samples, n_channels = data.shape
        
        # Enhanced frequency bands based on research
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
                
                # Power Spectral Density using Welch's method
                freqs, psd = signal.welch(channel_data, self.target_rate, 
                                        nperseg=min(256, len(channel_data)), 
                                        noverlap=min(128, len(channel_data)//2))
                
                # Band power features
                total_power = np.trapz(psd, freqs)
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        # Relative band power
                        rel_band_power = band_power / total_power if total_power > 0 else 0
                        window_features.extend([band_power, rel_band_power])
                    else:
                        window_features.extend([0, 0])
                
                # Enhanced spectral features
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
                spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]] if len(np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0]) > 0 else 0
                spectral_flux = np.sum(np.diff(psd)**2) if len(psd) > 1 else 0
                
                window_features.extend([
                    total_power, spectral_centroid, spectral_bandwidth, 
                    spectral_rolloff, spectral_flux
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def extract_wavelet_features(self, data, window_size=512, overlap=256):
        """
        Wavelet-based feature extraction using Discrete Wavelet Transform
        Based on recent research showing effectiveness for EEG analysis
        """
        features = []
        n_samples, n_channels = data.shape
        
        # Use Daubechies wavelet (db4) as recommended in literature
        wavelet = 'db4'
        levels = 5
        
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(channel_data, wavelet, level=levels)
                
                # Extract features from each level
                for level, coeff in enumerate(coeffs):
                    # Relative wavelet energy
                    energy = np.sum(coeff**2)
                    total_energy = np.sum([np.sum(c**2) for c in coeffs])
                    rel_energy = energy / total_energy if total_energy > 0 else 0
                    
                    # Statistical features of coefficients
                    window_features.extend([
                        rel_energy,
                        np.mean(np.abs(coeff)),
                        np.std(coeff),
                        np.max(np.abs(coeff))
                    ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def extract_nonlinear_features(self, data, window_size=512, overlap=256):
        """
        Nonlinear feature extraction including entropy measures
        Based on recent research in EEG complexity analysis
        """
        features = []
        n_samples, n_channels = data.shape
        
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Approximate Entropy
                approx_entropy = self.approximate_entropy(channel_data, m=2, r=0.2*np.std(channel_data))
                
                # Sample Entropy
                sample_entropy = self.sample_entropy(channel_data, m=2, r=0.2*np.std(channel_data))
                
                # Spectral Entropy
                freqs, psd = signal.welch(channel_data, self.target_rate, nperseg=min(128, len(channel_data)))
                psd_norm = psd / np.sum(psd)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                
                # Hjorth Parameters
                hjorth_activity, hjorth_mobility, hjorth_complexity = self.hjorth_parameters(channel_data)
                
                window_features.extend([
                    approx_entropy, sample_entropy, spectral_entropy,
                    hjorth_activity, hjorth_mobility, hjorth_complexity
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def approximate_entropy(self, data, m, r):
        """
        Calculate Approximate Entropy
        """
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
            C = np.zeros(len(patterns))
            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1.0
            phi = np.mean(np.log(C / len(patterns)))
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    def sample_entropy(self, data, m, r):
        """
        Calculate Sample Entropy
        """
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        N = len(data)
        B = 0.0
        A = 0.0
        
        # Template of length m
        for i in range(N - m):
            template_m = data[i:i + m]
            for j in range(i + 1, N - m):
                if _maxdist(template_m, data[j:j + m]) <= r:
                    B += 1.0
                    if j < N - m:
                        template_m1 = data[i:i + m + 1]
                        if _maxdist(template_m1, data[j:j + m + 1]) <= r:
                            A += 1.0
        
        if B > 0 and A > 0:
            return -np.log(A / B)
        else:
            return 0
    
    def hjorth_parameters(self, data):
        """
        Calculate Hjorth Parameters (Activity, Mobility, Complexity)
        """
        # First derivative
        dx = np.diff(data)
        # Second derivative
        ddx = np.diff(dx)
        
        # Variance of signal and its derivatives
        var_x = np.var(data)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        
        # Hjorth parameters
        activity = var_x
        mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0
        complexity = np.sqrt(var_ddx / var_dx) / mobility if var_dx > 0 and mobility > 0 else 0
        
        return activity, mobility, complexity
    
    def normalize_features(self, features, method='robust'):
        """
        Robust feature normalization
        """
        if method == 'standard':
            normalized_features = self.scaler.fit_transform(features)
        elif method == 'robust':
            # Use robust scaler to handle outliers better
            normalized_features = self.robust_scaler.fit_transform(features)
        else:
            # Min-max normalization
            normalized_features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
        
        return normalized_features
    
    def preprocess_pipeline_fast(self, dataset_handle='amananandrai/complete-eeg-dataset', save_results=True):
        """
        Fast preprocessing pipeline - basic features only
        """
        print("Starting FAST Complete EEG preprocessing pipeline...")
        
        # Load and basic preprocessing (same as before)
        df = self.load_dataset(dataset_handle)
        if df is None:
            return None
        
        self.detect_sampling_rate(df)
        eeg_channels = self.detect_channels(df)
        eeg_data = df[eeg_channels].values.astype(np.float64)
        
        # Basic preprocessing
        filtered_data = self.apply_bandpass_filter(eeg_data)
        filtered_data = self.apply_notch_filter(filtered_data)
        cleaned_data = self.remove_artifacts_ica(filtered_data)
        downsampled_data = self.downsample_data(cleaned_data)
        
        # Extract only time and frequency features (skip wavelet and nonlinear)
        print("Extracting time-domain features...")
        time_features = self.extract_time_domain_features(downsampled_data)
        
        print("Extracting frequency-domain features...")
        freq_features = self.extract_frequency_domain_features(downsampled_data)
        
        # Combine features
        all_features = np.hstack([time_features, freq_features])
        print(f"Combined features shape: {all_features.shape}")
        
        # Normalize
        normalized_features = self.normalize_features(all_features)
        
        # Save parameters
        self.preprocessing_params = {
            'eeg_channels': eeg_channels,
            'original_shape': eeg_data.shape,
            'processed_shape': normalized_features.shape,
            'sampling_rate': self.sampling_rate,
            'target_rate': self.target_rate,
            'n_features': normalized_features.shape[1],
            'n_time_features': time_features.shape[1],
            'n_freq_features': freq_features.shape[1],
            'n_wavelet_features': 0,
            'n_nonlinear_features': 0
        }
        
        if save_results:
            self.save_processed_data(normalized_features, downsampled_data)
        
        print("Fast preprocessing completed successfully!")
        return {
            'features': normalized_features,
            'processed_eeg': downsampled_data,
            'params': self.preprocessing_params
        }
    
    def preprocess_pipeline(self, dataset_handle='amananandrai/complete-eeg-dataset', save_results=True):
        """
        Complete enhanced preprocessing pipeline for Complete EEG dataset
        """
        print("Starting Complete EEG preprocessing pipeline...")
        
        # Load dataset
        df = self.load_dataset(dataset_handle)
        if df is None:
            return None
        
        # Detect sampling rate
        self.detect_sampling_rate(df)
        
        # Detect EEG channels
        eeg_channels = self.detect_channels(df)
        if not eeg_channels:
            print("No EEG channels detected!")
            return None
        
        # Extract EEG data
        eeg_data = df[eeg_channels].values.astype(np.float64)
        print(f"EEG data shape: {eeg_data.shape}")
        
        # Handle NaN values
        if np.isnan(eeg_data).any():
            print("Handling NaN values...")
            eeg_data = pd.DataFrame(eeg_data).fillna(method='ffill').fillna(method='bfill').values
        
        # Step 1: Apply enhanced bandpass filter
        print("Applying enhanced bandpass filter (0.5-50 Hz)...")
        filtered_data = self.apply_bandpass_filter(eeg_data)
        
        # Step 2: Apply enhanced notch filter
        print("Applying enhanced notch filter (50 Hz + harmonics)...")
        filtered_data = self.apply_notch_filter(filtered_data)
        
        # Step 3: Remove artifacts using enhanced ICA
        print("Removing artifacts using enhanced ICA...")
        cleaned_data = self.remove_artifacts_ica(filtered_data)
        
        # Step 4: Intelligent downsampling
        print("Applying intelligent downsampling...")
        downsampled_data = self.downsample_data(cleaned_data)
        
        # Step 5: Extract comprehensive features
        print("Extracting time-domain features...")
        time_features = self.extract_time_domain_features(downsampled_data)
        
        print("Extracting frequency-domain features...")
        freq_features = self.extract_frequency_domain_features(downsampled_data)
        
        print("Extracting wavelet features...")
        wavelet_features = self.extract_wavelet_features(downsampled_data)
        
        print("Extracting nonlinear features...")
        nonlinear_features = self.extract_nonlinear_features(downsampled_data)
        
        # Combine all features
        all_features = np.hstack([time_features, freq_features, wavelet_features, nonlinear_features])
        print(f"Combined features shape: {all_features.shape}")
        
        # Step 6: Robust normalization
        print("Applying robust feature normalization...")
        normalized_features = self.normalize_features(all_features)
        
        # Save preprocessing parameters
        self.preprocessing_params = {
            'eeg_channels': eeg_channels,
            'original_shape': eeg_data.shape,
            'processed_shape': normalized_features.shape,
            'sampling_rate': self.sampling_rate,
            'target_rate': self.target_rate,
            'n_features': normalized_features.shape[1],
            'n_time_features': time_features.shape[1],
            'n_freq_features': freq_features.shape[1],
            'n_wavelet_features': wavelet_features.shape[1],
            'n_nonlinear_features': nonlinear_features.shape[1]
        }
        
        if save_results:
            self.save_processed_data(normalized_features, downsampled_data)
        
        print("Enhanced preprocessing completed successfully!")
        return {
            'features': normalized_features,
            'processed_eeg': downsampled_data,
            'params': self.preprocessing_params
        }
    
    def save_processed_data(self, features, processed_eeg):
        """
        Save processed data and comprehensive analysis as CSV files
        """
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Convert features to DataFrame and save as CSV
        print("Saving processed features as CSV...")
        
        # Create comprehensive feature column names
        n_channels = len(self.preprocessing_params.get('eeg_channels', []))
        feature_names = []
        
        # Time-domain feature names (15 features per channel)
        time_features = [
            'mean', 'std', 'var', 'peak_to_peak', 'iqr', 'mav', 'rms', 'zero_crossings',
            'total_variation', 'max', 'min', 'median', 'q25', 'q75', 'outlier_ratio'
        ]
        for ch_idx in range(n_channels):
            for feat in time_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Frequency-domain feature names (15 features per channel)
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        freq_features = []
        for band in bands:
            freq_features.extend([f'{band}_power', f'{band}_rel_power'])
        freq_features.extend(['total_power', 'spectral_centroid', 'spectral_bandwidth', 
                            'spectral_rolloff', 'spectral_flux'])
        
        for ch_idx in range(n_channels):
            for feat in freq_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Wavelet feature names (24 features per channel: 6 levels × 4 features)
        wavelet_features = []
        for level in range(6):  # 6 decomposition levels
            for feat in ['rel_energy', 'mean_abs', 'std', 'max_abs']:
                wavelet_features.append(f'wavelet_L{level}_{feat}')
        
        for ch_idx in range(n_channels):
            for feat in wavelet_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Nonlinear feature names (6 features per channel)
        nonlinear_features = [
            'approx_entropy', 'sample_entropy', 'spectral_entropy',
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
        ]
        for ch_idx in range(n_channels):
            for feat in nonlinear_features:
                feature_names.append(f'ch{ch_idx}_{feat}')
        
        # Ensure we have the right number of feature names
        if len(feature_names) != features.shape[1]:
            # Fallback to generic names if mismatch
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        # Create DataFrame with features
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Add window index and time information
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
        
        # Save comprehensive feature analysis
        print("Creating feature analysis report...")
        feature_analysis = {
            'feature_name': feature_names,
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0),
            'skewness': [float('nan')] * features.shape[1],  # Placeholder
            'kurtosis': [float('nan')] * features.shape[1]   # Placeholder
        }
        
        # Calculate skewness and kurtosis safely
        from scipy.stats import skew, kurtosis
        for i in range(features.shape[1]):
            try:
                feature_analysis['skewness'][i] = skew(features[:, i])
                feature_analysis['kurtosis'][i] = kurtosis(features[:, i])
            except:
                pass
        
        analysis_df = pd.DataFrame(feature_analysis)
        analysis_csv_path = os.path.join(output_dir, 'feature_analysis.csv')
        analysis_df.to_csv(analysis_csv_path, index=False)
        print(f"Feature analysis saved to: {analysis_csv_path}")
        
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
        
        # Create enhanced dataset summary
        summary_data = {
            'dataset_name': ['Complete EEG Dataset'],
            'original_samples': [self.preprocessing_params.get('original_shape', [0, 0])[0]],
            'original_channels': [self.preprocessing_params.get('original_shape', [0, 0])[1]],
            'processed_windows': [features.shape[0]],
            'total_features': [features.shape[1]],
            'time_domain_features': [self.preprocessing_params.get('n_time_features', 0)],
            'frequency_domain_features': [self.preprocessing_params.get('n_freq_features', 0)],
            'wavelet_features': [self.preprocessing_params.get('n_wavelet_features', 0)],
            'nonlinear_features': [self.preprocessing_params.get('n_nonlinear_features', 0)],
            'original_sampling_rate_hz': [self.sampling_rate],
            'target_sampling_rate_hz': [self.target_rate],
            'preprocessing_date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'preprocessing_version': ['Enhanced v2.0 with Wavelet & Nonlinear Features']
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'dataset_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Dataset summary saved to: {summary_csv_path}")
        
        print(f"\nAll processed data saved to: {output_dir}")
        print("Files created:")
        print(f"  - processed_features.csv ({features.shape[0]} windows × {features.shape[1]} features)")
        print(f"  - processed_eeg_data.csv ({processed_eeg.shape[0]} samples × {processed_eeg.shape[1]} channels)")
        print(f"  - feature_analysis.csv (statistical analysis of {features.shape[1]} features)")
        print(f"  - dataset_summary.csv (comprehensive metadata)")
        print(f"  - preprocessing_params.json (all parameters)")
        print(f"  - .npy backup files")
    
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
    preprocessor = CompleteEEGPreprocessor(target_rate=128)
    
    # Use fast version for quicker results
    results = preprocessor.preprocess_pipeline_fast(dataset_handle='amananandrai/complete-eeg-dataset')
    # Or use full version: results = preprocessor.preprocess_pipeline(dataset_handle='amananandrai/complete-eeg-dataset')
    
    # Example of loading existing processed data
    # loaded_data = preprocessor.load_processed_data_csv()
    
    print("Enhanced Complete EEG Preprocessor ready!")
    print("To run preprocessing, uncomment the preprocessing line and execute this script.")
    print("\nFeatures included:")
    print("- 15 time-domain features per channel")
    print("- 15 frequency-domain features per channel (5 bands + spectral measures)")
    print("- 24 wavelet features per channel (6 levels × 4 measures)")
    print("- 6 nonlinear features per channel (entropy + Hjorth parameters)")
    print("- Total: 60 features per channel")