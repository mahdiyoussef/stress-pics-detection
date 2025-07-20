"""
Advanced EEG Signal Preprocessing Module

This module provides comprehensive preprocessing capabilities for EEG signals
used in stress detection. It includes filtering, artifact removal, feature extraction,
and normalization techniques based on current research best practices.

Key Features:
- Multi-band filtering (bandpass, notch)
- Artifact removal using ICA
- Time and frequency domain feature extraction
- Advanced normalization techniques
- Windowing and segmentation
- Data quality assessment

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import signal
from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import FastICA
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Advanced EEG Signal Preprocessor
    
    This class provides comprehensive preprocessing capabilities for EEG signals,
    specifically designed for stress detection applications. It implements
    state-of-the-art signal processing techniques and feature extraction methods.
    
    Attributes:
        sampling_rate (int): Original sampling rate of the EEG data
        target_rate (int): Target sampling rate after downsampling
        filter_params (Dict): Parameters for filtering operations
        feature_params (Dict): Parameters for feature extraction
        scaler (StandardScaler): Scaler for feature normalization
        ica (FastICA): ICA object for artifact removal
        preprocessing_params (Dict): Storage for preprocessing parameters
        
    Methods:
        preprocess_pipeline(): Complete preprocessing pipeline
        apply_filters(): Apply bandpass and notch filters
        remove_artifacts(): Remove artifacts using ICA
        extract_features(): Extract time and frequency domain features
        normalize_data(): Normalize features using various methods
    """
    
    def __init__(self, 
                 sampling_rate: int = 256, 
                 target_rate: int = 128,
                 filter_params: Optional[Dict] = None,
                 feature_params: Optional[Dict] = None):
        """
        Initialize the EEG Preprocessor.
        
        Args:
            sampling_rate (int): Original sampling rate in Hz
            target_rate (int): Target sampling rate after downsampling
            filter_params (Optional[Dict]): Custom filter parameters
            feature_params (Optional[Dict]): Custom feature extraction parameters
        """
        self.sampling_rate = sampling_rate
        self.target_rate = target_rate
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # ICA for artifact removal
        self.ica = None
        
        # Default filter parameters
        self.filter_params = filter_params or {
            'bandpass_low': 0.5,
            'bandpass_high': 50.0,
            'notch_freq': 50.0,
            'notch_quality': 30.0,
            'filter_order': 4
        }
        
        # Default feature extraction parameters
        self.feature_params = feature_params or {
            'window_size': 512,
            'overlap': 256,
            'frequency_bands': {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
        }
        
        # Storage for preprocessing parameters
        self.preprocessing_params = {}
        
        logger.info(f"EEG Preprocessor initialized: {sampling_rate}Hz -> {target_rate}Hz")
    
    def preprocess_pipeline(self, data: Union[np.ndarray, pd.DataFrame], 
                          apply_ica: bool = True,
                          extract_features: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for EEG data.
        
        This method applies the full preprocessing pipeline including filtering,
        artifact removal, feature extraction, and normalization.
        
        Args:
            data (Union[np.ndarray, pd.DataFrame]): Input EEG data
            apply_ica (bool): Whether to apply ICA for artifact removal
            extract_features (bool): Whether to extract features
            
        Returns:
            Dict[str, Any]: Dictionary containing processed data and metadata
        """
        logger.info("Starting EEG preprocessing pipeline...")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            eeg_channels = self.detect_channels(data)
            data_array = data[eeg_channels].values
        else:
            data_array = data
        
        # Validate input data
        self._validate_input_data(data_array)
        
        results = {}
        
        # Step 1: Apply filters
        logger.info("Step 1: Applying filters...")
        filtered_data = self.apply_filters(data_array)
        results['filtered_data'] = filtered_data
        
        # Step 2: Remove artifacts using ICA
        if apply_ica:
            logger.info("Step 2: Removing artifacts with ICA...")
            cleaned_data = self.remove_artifacts_ica(filtered_data)
        else:
            cleaned_data = filtered_data
        results['cleaned_data'] = cleaned_data
        
        # Step 3: Downsample data
        logger.info("Step 3: Downsampling data...")
        downsampled_data = self.downsample_data(cleaned_data)
        results['downsampled_data'] = downsampled_data
        
        # Step 4: Extract features
        if extract_features:
            logger.info("Step 4: Extracting features...")
            
            # Time domain features
            time_features = self.extract_time_domain_features(downsampled_data)
            
            # Frequency domain features
            freq_features = self.extract_frequency_domain_features(downsampled_data)
            
            # Combine features
            if time_features.size > 0 and freq_features.size > 0:
                combined_features = np.hstack([time_features, freq_features])
            elif time_features.size > 0:
                combined_features = time_features
            else:
                combined_features = freq_features
            
            # Normalize features
            normalized_features = self.normalize_features(combined_features)
            
            results['time_features'] = time_features
            results['frequency_features'] = freq_features
            results['combined_features'] = combined_features
            results['normalized_features'] = normalized_features
        
        # Store preprocessing parameters
        results['preprocessing_params'] = self.preprocessing_params
        results['data_shape'] = data_array.shape
        results['processed_shape'] = downsampled_data.shape
        
        logger.info("Preprocessing pipeline completed successfully!")
        return results
    
    def detect_channels(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect EEG channels in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            List[str]: List of detected EEG channel names
        """
        # Common EEG channel names based on 10-20 system
        eeg_patterns = [
            'EEG', 'eeg', 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 
            'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz', 'AF3', 'AF4', 'FC1', 'FC2', 'CP1', 'CP2',
            'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
        ]
        
        eeg_channels = []
        for col in df.columns:
            if any(pattern in col for pattern in eeg_patterns):
                eeg_channels.append(col)
        
        # If no standard names found, assume numeric columns are EEG
        if not eeg_channels:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove obvious non-EEG columns
            exclude_patterns = ['time', 'timestamp', 'label', 'target', 'class', 'id', 'index']
            eeg_channels = [col for col in numeric_cols 
                          if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        logger.info(f"Detected {len(eeg_channels)} EEG channels")
        return eeg_channels
    
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters to remove noise and artifacts.
        
        Args:
            data (np.ndarray): Input EEG data (samples x channels)
            
        Returns:
            np.ndarray: Filtered EEG data
        """
        # Apply bandpass filter
        filtered_data = self.apply_bandpass_filter(
            data, 
            self.filter_params['bandpass_low'],
            self.filter_params['bandpass_high']
        )
        
        # Apply notch filter for powerline interference
        filtered_data = self.apply_notch_filter(
            filtered_data,
            self.filter_params['notch_freq'],
            self.filter_params['notch_quality']
        )
        
        return filtered_data
    
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low_freq: float = 0.5, 
                            high_freq: float = 50.0) -> np.ndarray:
        """
        Apply bandpass filter to remove artifacts and noise.
        
        Args:
            data (np.ndarray): Input data
            low_freq (float): Low cutoff frequency
            high_freq (float): High cutoff frequency
            
        Returns:
            np.ndarray: Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        try:
            # Design Butterworth bandpass filter
            b, a = signal.butter(
                self.filter_params['filter_order'], 
                [low, high], 
                btype='band'
            )
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            
            logger.info(f"Applied bandpass filter: {low_freq}-{high_freq} Hz")
            return filtered_data
            
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}. Returning original data.")
            return data
    
    def apply_notch_filter(self, data: np.ndarray, 
                         notch_freq: float = 50.0, 
                         quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            data (np.ndarray): Input data
            notch_freq (float): Notch frequency (typically 50 or 60 Hz)
            quality_factor (float): Quality factor of the notch filter
            
        Returns:
            np.ndarray: Filtered data
        """
        try:
            # Design notch filter
            b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            
            logger.info(f"Applied notch filter at {notch_freq} Hz")
            return filtered_data
            
        except Exception as e:
            logger.warning(f"Notch filter failed: {e}. Returning original data.")
            return data
    
    def remove_artifacts_ica(self, data: np.ndarray, 
                           n_components: Optional[int] = None,
                           artifact_threshold: float = 3.0) -> np.ndarray:
        """
        Remove artifacts using Independent Component Analysis (ICA).
        
        Args:
            data (np.ndarray): Input data
            n_components (Optional[int]): Number of ICA components
            artifact_threshold (float): Threshold for artifact detection
            
        Returns:
            np.ndarray: Cleaned data
        """
        if n_components is None:
            n_components = min(data.shape[1], 15)
        
        try:
            # Apply ICA
            self.ica = FastICA(
                n_components=n_components, 
                random_state=42, 
                max_iter=1000,
                tol=1e-4
            )
            
            # Fit and transform
            ica_components = self.ica.fit_transform(data.T).T
            
            # Automatic artifact detection based on statistical properties
            clean_components = []
            for i, component in enumerate(ica_components.T):
                # Check for extreme values (artifacts typically have high kurtosis)
                component_kurtosis = kurtosis(component)
                component_max_z = np.abs(zscore(component)).max()
                
                # Keep components that don't show artifact characteristics
                if (component_max_z < artifact_threshold and 
                    abs(component_kurtosis) < 10):  # Normal EEG has kurtosis < 10
                    clean_components.append(i)
            
            # Reconstruct data using only clean components
            if clean_components:
                clean_ica = np.zeros_like(ica_components)
                clean_ica[:, clean_components] = ica_components[:, clean_components]
                cleaned_data = self.ica.inverse_transform(clean_ica.T).T
            else:
                logger.warning("No clean components found, keeping original data")
                cleaned_data = data
            
            removed_components = n_components - len(clean_components)
            logger.info(f"ICA: Removed {removed_components} artifact components")
            
            return cleaned_data
            
        except Exception as e:
            logger.warning(f"ICA failed: {e}. Returning original data.")
            return data
    
    def downsample_data(self, data: np.ndarray) -> np.ndarray:
        """
        Downsample data to target sampling rate.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Downsampled data
        """
        if self.target_rate >= self.sampling_rate:
            return data
        
        # Calculate downsampling factor
        downsample_factor = self.sampling_rate // self.target_rate
        
        # Apply anti-aliasing filter before downsampling
        nyquist = self.target_rate / 2
        cutoff = nyquist / (self.sampling_rate / 2)
        
        try:
            # Design anti-aliasing filter
            b, a = signal.butter(4, cutoff, btype='low')
            
            # Apply filter and downsample
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            
            # Downsample
            downsampled_data = filtered_data[::downsample_factor, :]
            
            logger.info(f"Downsampled from {self.sampling_rate}Hz to {self.target_rate}Hz")
            return downsampled_data
            
        except Exception as e:
            logger.warning(f"Downsampling failed: {e}. Using simple decimation.")
            return data[::downsample_factor, :]
    
    def extract_time_domain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive time-domain features from EEG data.
        
        Args:
            data (np.ndarray): Input EEG data
            
        Returns:
            np.ndarray: Extracted time-domain features
        """
        features = []
        n_samples, n_channels = data.shape
        window_size = self.feature_params['window_size']
        overlap = self.feature_params['overlap']
        
        # Sliding window approach
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Statistical features
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                var_val = np.var(channel_data)
                peak_to_peak = np.max(channel_data) - np.min(channel_data)
                iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
                mean_abs = np.mean(np.abs(channel_data))
                rms = np.sqrt(np.mean(channel_data**2))
                
                # Zero crossings
                zero_crossings = len(np.where(np.diff(np.sign(channel_data)))[0])
                
                # Higher order statistics
                skewness = skew(channel_data)
                kurt = kurtosis(channel_data)
                
                # Energy and complexity measures
                energy = np.sum(channel_data**2)
                complexity = np.sum(np.abs(np.diff(channel_data)))
                
                window_features.extend([
                    mean_val, std_val, var_val, peak_to_peak, iqr,
                    mean_abs, rms, zero_crossings, skewness, kurt,
                    energy, complexity
                ])
            
            features.append(window_features)
        
        logger.info(f"Extracted {len(features)} time-domain feature windows")
        return np.array(features)
    
    def extract_frequency_domain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive frequency-domain features from EEG data.
        
        Args:
            data (np.ndarray): Input EEG data
            
        Returns:
            np.ndarray: Extracted frequency-domain features
        """
        features = []
        n_samples, n_channels = data.shape
        window_size = self.feature_params['window_size']
        overlap = self.feature_params['overlap']
        bands = self.feature_params['frequency_bands']
        
        # Sliding window approach
        for start in range(0, n_samples - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = data[start:end, :]
            
            window_features = []
            for ch in range(n_channels):
                channel_data = window_data[:, ch]
                
                # Power Spectral Density
                freqs, psd = signal.welch(
                    channel_data, 
                    self.target_rate, 
                    nperseg=min(256, len(channel_data)),
                    noverlap=min(128, len(channel_data)//2)
                )
                
                # Band power features
                total_power = np.trapz(psd, freqs)
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        relative_power = band_power / total_power if total_power > 0 else 0
                        window_features.extend([band_power, relative_power])
                    else:
                        window_features.extend([0, 0])
                
                # Spectral features
                if np.sum(psd) > 0:
                    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                    spectral_bandwidth = np.sqrt(
                        np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)
                    )
                    spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
                else:
                    spectral_centroid = spectral_bandwidth = spectral_rolloff = 0
                
                # Peak frequency
                peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
                
                window_features.extend([
                    total_power, spectral_centroid, spectral_bandwidth,
                    spectral_rolloff, peak_freq
                ])
            
            features.append(window_features)
        
        logger.info(f"Extracted {len(features)} frequency-domain feature windows")
        return np.array(features)
    
    def normalize_features(self, features: np.ndarray, 
                         method: str = 'standard') -> np.ndarray:
        """
        Normalize extracted features using various methods.
        
        Args:
            features (np.ndarray): Input features
            method (str): Normalization method ('standard', 'robust', 'minmax')
            
        Returns:
            np.ndarray: Normalized features
        """
        if features.size == 0:
            return features
        
        try:
            if method == 'standard':
                normalized_features = self.scaler.fit_transform(features)
            elif method == 'robust':
                normalized_features = self.robust_scaler.fit_transform(features)
            elif method == 'minmax':
                normalized_features = self.minmax_scaler.fit_transform(features)
            else:
                logger.warning(f"Unknown normalization method: {method}. Using standard.")
                normalized_features = self.scaler.fit_transform(features)
            
            logger.info(f"Features normalized using {method} method")
            return normalized_features
            
        except Exception as e:
            logger.warning(f"Normalization failed: {e}. Returning original features.")
            return features
    
    def _validate_input_data(self, data: np.ndarray) -> None:
        """
        Validate input data format and quality.
        
        Args:
            data (np.ndarray): Input data to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        if data.size == 0:
            raise ValueError("Input data is empty")
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data (samples x channels), got shape {data.shape}")
        
        # Check for infinite or NaN values
        if not np.isfinite(data).all():
            logger.warning("Data contains infinite or NaN values. Consider data cleaning.")
        
        # Check data range (EEG typically in microvolts, should be reasonable)
        data_range = np.max(data) - np.min(data)
        if data_range > 1e6:  # Very large range might indicate scaling issues
            logger.warning(f"Large data range detected: {data_range}. Check data scaling.")
        
        logger.info(f"Data validation passed: {data.shape}")
    
    def save_preprocessing_params(self, filepath: str) -> None:
        """
        Save preprocessing parameters to file.
        
        Args:
            filepath (str): Path to save parameters
        """
        import json
        
        params = {
            'sampling_rate': self.sampling_rate,
            'target_rate': self.target_rate,
            'filter_params': self.filter_params,
            'feature_params': self.feature_params,
            'preprocessing_params': self.preprocessing_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Preprocessing parameters saved to {filepath}")
    
    def load_preprocessing_params(self, filepath: str) -> None:
        """
        Load preprocessing parameters from file.
        
        Args:
            filepath (str): Path to load parameters from
        """
        import json
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.sampling_rate = params.get('sampling_rate', self.sampling_rate)
        self.target_rate = params.get('target_rate', self.target_rate)
        self.filter_params.update(params.get('filter_params', {}))
        self.feature_params.update(params.get('feature_params', {}))
        self.preprocessing_params = params.get('preprocessing_params', {})
        
        logger.info(f"Preprocessing parameters loaded from {filepath}")


# Legacy class for backward compatibility
class EEGMentalStatePreprocessor(EEGPreprocessor):
    """Legacy class for backward compatibility."""
    pass


if __name__ == "__main__":
    # Example usage
    preprocessor = EEGPreprocessor(sampling_rate=256, target_rate=128)
    
    # Generate sample data for testing
    n_samples, n_channels = 1000, 4
    sample_data = np.random.randn(n_samples, n_channels)
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Processed data shape: {results['processed_shape']}")
    if 'normalized_features' in results:
        print(f"Features shape: {results['normalized_features'].shape}")
    
    print("Preprocessing completed successfully!")
