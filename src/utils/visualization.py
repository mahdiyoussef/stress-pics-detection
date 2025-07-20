"""
Visualization Utilities for EEG Stress Detection Project

This module provides comprehensive visualization tools for EEG data analysis,
model performance evaluation, and result presentation.

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EEGVisualizer:
    """
    Comprehensive visualization toolkit for EEG data and model results.
    
    This class provides methods for visualizing:
    - Raw and preprocessed EEG signals
    - Feature distributions and correlations
    - Model training history
    - Performance metrics and confusion matrices
    - Stress level predictions over time
    """
    
    def __init__(self, save_dir: str = 'results/plots'):
        """
        Initialize the EEG Visualizer.
        
        Args:
            save_dir (str): Directory to save generated plots
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set up plotting parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        logger.info("EEG Visualizer initialized")
    
    def plot_eeg_signals(self, 
                        data: np.ndarray, 
                        channels: Optional[List[str]] = None,
                        sampling_rate: int = 256,
                        duration: Optional[float] = None,
                        title: str = "EEG Signals") -> None:
        """
        Plot raw EEG signals for multiple channels.
        
        Args:
            data (np.ndarray): EEG data (samples x channels)
            channels (Optional[List[str]]): Channel names
            sampling_rate (int): Sampling rate in Hz
            duration (Optional[float]): Duration to plot in seconds
            title (str): Plot title
        """
        if duration is not None:
            n_samples = int(duration * sampling_rate)
            data = data[:n_samples, :]
        
        n_samples, n_channels = data.shape
        time_axis = np.linspace(0, n_samples / sampling_rate, n_samples)
        
        if channels is None:
            channels = [f'Channel {i+1}' for i in range(n_channels)]
        
        # Create subplots
        fig, axes = plt.subplots(min(n_channels, 8), 1, figsize=(15, 2*min(n_channels, 8)))
        if n_channels == 1:
            axes = [axes]
        
        for i in range(min(n_channels, 8)):
            axes[i].plot(time_axis, data[:, i], linewidth=0.8, alpha=0.8)
            axes[i].set_ylabel(f'{channels[i]} (μV)')
            axes[i].set_xlim(0, time_axis[-1])
            
            if i == len(axes) - 1:
                axes[i].set_xlabel('Time (s)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/eeg_signals_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distributions(self, 
                                 features: np.ndarray, 
                                 labels: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 max_features: int = 16) -> None:
        """
        Plot distributions of extracted features by stress level.
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Stress labels
            feature_names (Optional[List[str]]): Names of features
            max_features (int): Maximum number of features to plot
        """
        n_features = min(features.shape[1], max_features)
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Create subplot grid
        cols = 4
        rows = (n_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
        
        stress_levels = ['Low', 'Medium', 'High']
        colors = ['green', 'orange', 'red']
        
        for i in range(n_features):
            for level, color in zip(range(3), colors):
                mask = labels == level
                if np.any(mask):
                    axes[i].hist(features[mask, i], alpha=0.6, 
                               label=f'{stress_levels[level]} Stress',
                               color=color, bins=20)
            
            axes[i].set_title(feature_names[i])
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Feature Distributions by Stress Level', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/feature_distributions_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, 
                              features: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              max_features: int = 50) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            features (np.ndarray): Feature matrix
            feature_names (Optional[List[str]]): Names of features
            max_features (int): Maximum number of features to include
        """
        n_features = min(features.shape[1], max_features)
        features_subset = features[:, :n_features]
        
        if feature_names is None:
            feature_names = [f'F{i+1}' for i in range(n_features)]
        else:
            feature_names = feature_names[:n_features]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features_subset.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, xticklabels=feature_names, yticklabels=feature_names,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/correlation_matrix_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            title: str = "Training History") -> None:
        """
        Plot comprehensive training history.
        
        Args:
            history (Dict[str, List[float]]): Training history dictionary
            title (str): Plot title
        """
        metrics = ['accuracy', 'loss', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in history]
        
        if not available_metrics:
            logger.warning("No metrics found in history")
            return
        
        n_metrics = len(available_metrics)
        cols = 2
        rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            epochs = range(1, len(history[metric]) + 1)
            
            axes[i].plot(epochs, history[metric], 'b-', label=f'Training {metric.title()}', linewidth=2)
            
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[i].plot(epochs, history[val_metric], 'r-', 
                           label=f'Validation {metric.title()}', linewidth=2)
            
            axes[i].set_title(f'{metric.title()} over Epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/training_history_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray, 
                            class_names: List[str],
                            title: str = "Confusion Matrix",
                            normalize: bool = False) -> None:
        """
        Plot confusion matrix with enhanced visualization.
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_names (List[str]): Class names
            title (str): Plot title
            normalize (bool): Whether to normalize the matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage' if normalize else 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy information
        if not normalize:
            accuracy = np.trace(cm) / np.sum(cm)
            plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                       fontsize=10, ha='left')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/confusion_matrix_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stress_predictions(self, 
                              predictions: np.ndarray,
                              true_labels: Optional[np.ndarray] = None,
                              time_axis: Optional[np.ndarray] = None,
                              title: str = "Stress Level Predictions") -> None:
        """
        Plot stress level predictions over time.
        
        Args:
            predictions (np.ndarray): Predicted stress levels
            true_labels (Optional[np.ndarray]): True stress levels
            time_axis (Optional[np.ndarray]): Time axis for plotting
            title (str): Plot title
        """
        if time_axis is None:
            time_axis = np.arange(len(predictions))
        
        plt.figure(figsize=(15, 6))
        
        # Plot predictions
        stress_colors = {0: 'green', 1: 'orange', 2: 'red'}
        stress_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        for level in range(3):
            mask = predictions == level
            if np.any(mask):
                plt.scatter(time_axis[mask], predictions[mask], 
                          c=stress_colors[level], label=f'{stress_names[level]} Stress',
                          alpha=0.7, s=20)
        
        # Plot true labels if available
        if true_labels is not None:
            plt.plot(time_axis, true_labels, 'k-', alpha=0.3, linewidth=1, 
                    label='True Labels')
        
        plt.xlabel('Time Index')
        plt.ylabel('Stress Level')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/stress_predictions_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_architecture(self, 
                              model_summary: str,
                              title: str = "Model Architecture") -> None:
        """
        Create a visual representation of model architecture.
        
        Args:
            model_summary (str): Model summary string
            title (str): Plot title
        """
        # This is a simplified visualization
        # For more complex visualizations, consider using tools like Netron
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, model_summary, transform=ax.transAxes,
                fontsize=8, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=1", 
                facecolor="lightgray", alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/model_architecture_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_dashboard(self, 
                                   metrics: Dict[str, float],
                                   cm: np.ndarray,
                                   history: Dict[str, List[float]],
                                   class_names: List[str]) -> None:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            metrics (Dict[str, float]): Performance metrics
            cm (np.ndarray): Confusion matrix
            history (Dict[str, List[float]]): Training history
            class_names (List[str]): Class names
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Metrics summary
        ax1 = plt.subplot(2, 3, 1)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrix
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Training history - Accuracy
        ax3 = plt.subplot(2, 3, 3)
        epochs = range(1, len(history['accuracy']) + 1)
        ax3.plot(epochs, history['accuracy'], 'b-', label='Training', linewidth=2)
        if 'val_accuracy' in history:
            ax3.plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training history - Loss
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
        if 'val_loss' in history:
            ax4.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax4.set_title('Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Class distribution
        ax5 = plt.subplot(2, 3, 5)
        class_counts = np.sum(cm, axis=1)
        ax5.pie(class_counts, labels=class_names, autopct='%1.1f%%',
               colors=['lightgreen', 'orange', 'lightcoral'])
        ax5.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Model performance summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        Model Performance Summary
        
        Overall Accuracy: {metrics.get('accuracy', 0):.3f}
        F1-Score: {metrics.get('f1_score', 0):.3f}
        Precision: {metrics.get('precision', 0):.3f}
        Recall: {metrics.get('recall', 0):.3f}
        
        Total Samples: {np.sum(cm)}
        Training Epochs: {len(history['accuracy'])}
        
        Best Performance:
        - Accuracy: {max(history.get('val_accuracy', [0])):.3f}
        - Min Loss: {min(history.get('val_loss', [float('inf')])):.3f}
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('EEG Stress Detection Model Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/performance_dashboard_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


def create_interactive_plot(data: np.ndarray, 
                          title: str = "Interactive EEG Plot") -> None:
    """
    Create an interactive plot using Plotly.
    
    Args:
        data (np.ndarray): Data to plot
        title (str): Plot title
    """
    fig = go.Figure()
    
    for i in range(min(data.shape[1], 8)):  # Limit to 8 channels
        fig.add_trace(go.Scatter(
            y=data[:, i],
            mode='lines',
            name=f'Channel {i+1}',
            line=dict(width=1)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sample Index',
        yaxis_title='Amplitude (μV)',
        hovermode='x unified',
        height=600
    )
    
    fig.show()


if __name__ == "__main__":
    # Example usage
    visualizer = EEGVisualizer()
    
    # Generate sample data
    n_samples, n_channels = 1000, 4
    sample_data = np.random.randn(n_samples, n_channels) * 50
    
    # Plot sample EEG signals
    visualizer.plot_eeg_signals(sample_data, duration=4.0)
    
    print("Visualization examples completed!")
