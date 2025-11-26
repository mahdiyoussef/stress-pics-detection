# Predicting Stress Peaks based on EEG Signals by Hybrid Approach of CNN-LSTM

## Overview

This project implements an advanced deep learning approach for predicting stress peaks in EEG (Electroencephalogram) signals using a hybrid CNN-LSTM neural network architecture. The system combines Convolutional Neural Networks (CNN) for local feature extraction and Long Short-Term Memory (LSTM) networks for temporal pattern recognition to achieve high accuracy in stress detection.

## Objectives

- **Primary Goal**: Develop an accurate stress detection system using EEG brainwave data
- **Secondary Goals**: 
  - Implement advanced signal preprocessing techniques
  - Create a robust hybrid CNN-LSTM model
  - Achieve high accuracy in multi-class stress classification
  - Provide comprehensive evaluation metrics and visualizations

## Project Structure

```
Predicting-Stress-Pics-based-on-EEG-Signals-by-Hybrid-Approach-of-CNN-LSTM/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # License information
├── .gitignore                        # Git ignore file
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Dataset loading utilities
│   │   └── data_augmentation.py     # Data augmentation techniques
│   │
│   ├── preprocessing/               # Data preprocessing
│   │   ├── __init__.py
│   │   ├── eeg_preprocessor.py      # EEG signal preprocessing
│   │   └── feature_extraction.py   # Feature extraction methods
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn_lstm_model.py       # Hybrid CNN-LSTM model
│   │   ├── base_model.py           # Base model class
│   │   └── model_utils.py          # Model utilities
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── visualization.py        # Plotting and visualization
│       ├── metrics.py              # Evaluation metrics
│       └── config.py               # Configuration settings
│
├── data/                            # Data storage
│   ├── raw/                        # Raw datasets
│   └── processed/                  # Processed datasets
│
├── models/                          # Trained models storage
│   ├── saved_models/               # Model checkpoints
│   └── best_models/                # Best performing models
│
├── results/                         # Results and outputs
│   ├── plots/                      # Generated plots
│   ├── metrics/                    # Performance metrics
│   └── logs/                       # Training logs
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Data exploration
│   ├── 02_preprocessing.ipynb      # Preprocessing analysis
│   ├── 03_model_training.ipynb     # Model training
│   └── 04_results_analysis.ipynb   # Results analysis
│
└── docs/                           # Documentation
    ├── methodology.md              # Detailed methodology
    ├── architecture.md             # Model architecture details
    └── api_reference.md            # API reference
```

## Dataset Information

The project utilizes multiple EEG datasets:

1. **EEG Mental State Dataset**: Primary dataset for stress detection
2. **EEG Emotions Dataset**: Supplementary emotional state data
3. **Complete EEG Dataset**: Comprehensive EEG recordings
4. **General EEG Dataset**: Additional training data

### Dataset Features:
- **Channels**: Multiple EEG channels (typically 14-64 channels)
- **Sampling Rate**: 128-256 Hz
- **Duration**: Variable recording lengths
- **Labels**: Stress levels (Low, Medium, High)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Predicting-Stress-Pics-based-on-EEG-Signals-by-Hybrid-Approach-of-CNN-LSTM.git
cd Predicting-Stress-Pics-based-on-EEG-Signals-by-Hybrid-Approach-of-CNN-LSTM
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install package in development mode**:
```bash
pip install -e .
```

## Quick Start

### 1. Data Preparation
```python
from src.data.data_loader import EEGDataLoader
from src.preprocessing.eeg_preprocessor import EEGPreprocessor

# Load datasets
loader = EEGDataLoader()
raw_data = loader.load_all_datasets()

# Preprocess data
preprocessor = EEGPreprocessor()
processed_data = preprocessor.preprocess_pipeline(raw_data)
```

### 2. Model Training
```python
from src.models.cnn_lstm_model import CNNLSTMStressDetector

# Initialize model
model = CNNLSTMStressDetector()

# Train model
history = model.train(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100
)
```

### 3. Evaluation
```python
# Evaluate model
results = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

## Model Architecture

### Hybrid CNN-LSTM Architecture

The model combines:

1. **Convolutional Layers** (Feature Extraction):
   - Multi-scale 1D convolutions
   - Batch normalization
   - Dropout for regularization

2. **LSTM Layers** (Temporal Modeling):
   - Bidirectional LSTM
   - Attention mechanisms
   - Dropout and recurrent dropout

3. **Dense Layers** (Classification):
   - Fully connected layers
   - Advanced regularization
   - Multi-class output

### Key Features:
- **Multi-scale Feature Extraction**: Different kernel sizes for various frequency components
- **Temporal Dependencies**: LSTM captures long-term temporal patterns
- **Attention Mechanism**: Focuses on relevant time periods
- **Advanced Regularization**: Prevents overfitting with small datasets
- **Data Augmentation**: Synthetic data generation for robust training

## Performance Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for imbalanced classes
- **Precision**: Class-wise precision
- **Recall**: Class-wise recall
- **Confusion Matrix**: Detailed classification results
- **ROC Curves**: Receiver Operating Characteristic analysis

### Typical Performance:
- **Accuracy**: 85-92%
- **F1-Score**: 0.83-0.90
- **Training Time**: 30-60 minutes on GPU

## Key Features

### Advanced Preprocessing:
- **Bandpass Filtering**: Removes noise and artifacts
- **ICA Denoising**: Independent Component Analysis for artifact removal
- **Feature Engineering**: Time and frequency domain features
- **Normalization**: Robust scaling for stable training

### Data Augmentation:
- **Noise Injection**: Gaussian noise addition
- **Time Shifting**: Temporal signal shifting
- **Amplitude Scaling**: Signal amplitude variations
- **Frequency Domain Augmentation**: Spectral modifications

### Model Enhancements:
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Model Checkpointing**: Saves best models
- **Cross-Validation**: Robust model evaluation

## Results and Visualizations

The project generates comprehensive visualizations:

1. **Training History Plots**: Accuracy and loss curves
2. **Confusion Matrices**: Classification performance
3. **Feature Importance**: Most relevant EEG features
4. **Signal Visualizations**: Raw and processed EEG signals
5. **ROC Curves**: Model performance analysis

## Configuration

Key configuration parameters in `src/utils/config.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    'cnn_filters': [32, 64, 128],
    'cnn_kernels': [3, 5, 7],
    'lstm_units': [64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'sampling_rate': 256,
    'target_rate': 128,
    'bandpass_low': 0.5,
    'bandpass_high': 50,
    'window_size': 1000
}
```

## Usage Examples

### Example 1: Basic Training
```python
from src.models.cnn_lstm_model import CNNLSTMStressDetector

# Initialize and train
detector = CNNLSTMStressDetector()
detector.train_full_pipeline()
```

### Example 2: Custom Preprocessing
```python
from src.preprocessing.eeg_preprocessor import EEGPreprocessor

preprocessor = EEGPreprocessor(
    sampling_rate=256,
    target_rate=128,
    apply_ica=True
)
processed_data = preprocessor.process_dataset('path/to/data')
```

### Example 3: Model Evaluation
```python
# Load trained model
detector = CNNLSTMStressDetector.load_model('models/best_model.h5')

# Evaluate on test set
metrics = detector.comprehensive_evaluation(X_test, y_test)
```

## Research Methodology

### Signal Processing Pipeline:
1. **Data Loading**: Import multiple EEG datasets
2. **Quality Assessment**: Signal quality evaluation
3. **Preprocessing**: Filtering, denoising, normalization
4. **Feature Extraction**: Time and frequency domain features
5. **Augmentation**: Data augmentation for robustness
6. **Model Training**: Hybrid CNN-LSTM training
7. **Evaluation**: Comprehensive performance assessment

### Validation Strategy:
- **Train/Validation/Test Split**: 70/15/15 split
- **Cross-Validation**: 5-fold stratified cross-validation
- **Independent Test Set**: Final evaluation on unseen data

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup:
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: Thanks to the providers of EEG datasets on Kaggle
- **Research**: Based on recent advances in EEG signal processing and deep learning
- **Libraries**: TensorFlow, Scikit-learn, SciPy, and other open-source libraries


## Citation

If you use this project in your research, please cite:

```bibtex
@article{mahdi2024stress,
  title={Predicting Stress Peaks based on EEG Signals by Hybrid Approach of CNN-LSTM},
  author={Mahdi, Youssef and El Haiki, Hamza},
  year={2024},
  journal={Your Journal},
  volume={XX},
  pages={XXX-XXX}
}
```

## Version History

- **v1.0.0** (2024-07-20): Initial release
- **v1.1.0** (TBD): Enhanced model architecture
- **v1.2.0** (TBD): Additional datasets integration

---

**Note**: This project is for research and educational purposes. For medical applications, please consult with healthcare professionals and ensure proper validation.
