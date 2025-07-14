# Install kagglehub
# pip install kagglehub pandas numpy

import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# Authenticate (required for first time)
# kagglehub.login()  # Uncomment this for first time setup

# Alternative: Set environment variables
# export KAGGLE_USERNAME=your_username
# export KAGGLE_KEY=your_api_key

# Download EEG datasets
def download_eeg_datasets():
    datasets = {
        'eeg_mental_state': 'birdy654/eeg-brainwave-dataset-mental-state',
        'eeg_emotions': 'birdy654/eeg-brainwave-dataset-feeling-emotions', 
        'eeg_general': 'samnikolas/eeg-dataset',
        'complete_eeg': 'amananandrai/complete-eeg-dataset'
    }
    
    downloaded_paths = {}
    
    for name, dataset_handle in datasets.items():
        try:
            print(f"Downloading {name}...")
            path = kagglehub.dataset_download(dataset_handle)
            downloaded_paths[name] = path
            print(f"Downloaded {name} to: {path}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    return downloaded_paths

# Load dataset as pandas DataFrame
def load_eeg_as_dataframe(dataset_handle, filename=None):
    try:
        if filename:
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                dataset_handle,
                filename
            )
        else:
            # Download entire dataset
            path = kagglehub.dataset_download(dataset_handle)
            print(f"Dataset downloaded to: {path}")
            return path
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_dataset_info():
    """Return information about available EEG datasets"""
    return {
        'eeg_mental_state': {
            'handle': 'birdy654/eeg-brainwave-dataset-mental-state',
            'description': 'EEG brainwave data for mental state classification',
            'suitable_for': 'stress detection, mental state analysis'
        },
        'eeg_emotions': {
            'handle': 'birdy654/eeg-brainwave-dataset-feeling-emotions',
            'description': 'EEG data for emotion recognition',
            'suitable_for': 'emotion detection, stress analysis'
        },
        'eeg_general': {
            'handle': 'samnikolas/eeg-dataset',
            'description': 'General EEG dataset',
            'suitable_for': 'general EEG analysis'
        },
        'complete_eeg': {
            'handle': 'amananandrai/complete-eeg-dataset',
            'description': 'Comprehensive EEG dataset',
            'suitable_for': 'comprehensive EEG analysis'
        }
    }

# Example usage
if __name__ == "__main__":
    # Download all EEG datasets
    paths = download_eeg_datasets()
    
    # Print dataset info
    info = get_dataset_info()
    for name, details in info.items():
        print(f"\n{name}: {details['description']}")
    
    # Load specific dataset as DataFrame (if CSV available)
    # df = load_eeg_as_dataframe('birdy654/eeg-brainwave-dataset-mental-state', 'data.csv')