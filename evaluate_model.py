"""
Model Evaluation Script for EEG Stress Detection

This script provides comprehensive evaluation capabilities for trained models,
including performance metrics, visualization, and comparison tools.

Usage:
    python evaluate_model.py --model MODEL_PATH [--data DATA_PATH] [--output OUTPUT_DIR]

Author: Youssef Mahdi, Hamza El Haiki
Date: July 2024
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import EEGDataLoader
from src.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.models.cnn_lstm_model import CNNLSTMStressDetector
from src.utils.config import (
    MODEL_CONFIG, PREPROCESSING_CONFIG, RESULTS_DIR, 
    create_directories, get_config
)
from src.utils.visualization import EEGVisualizer


class ModelEvaluator:
    """
    Comprehensive model evaluation class for EEG stress detection models.
    """
    
    def __init__(self, model_path: str, output_dir: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to the trained model
            output_dir (str): Directory to save evaluation results
        """
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join(RESULTS_DIR, 'evaluation')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.model = self._load_model()
        self.visualizer = EEGVisualizer()
        
        # Class names
        self.class_names = MODEL_CONFIG['output']['class_names']
        self.num_classes = MODEL_CONFIG['output']['num_classes']
    
    def _setup_logging(self) -> None:
        """Setup logging for evaluation."""
        log_file = os.path.join(self.output_dir, f'evaluation_{self.timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('ModelEvaluator')
        self.logger.info(f"Model Evaluator initialized")
        self.logger.info(f"Model path: {self.model_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _load_model(self) -> CNNLSTMStressDetector:
        """Load the trained model."""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Initialize model with same configuration
            model = CNNLSTMStressDetector(**MODEL_CONFIG['cnn_lstm'])
            
            # Load weights
            model.load_model(self.model_path)
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_test_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data for evaluation.
        
        Args:
            data_path (str): Path to test data (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and labels
        """
        if data_path and os.path.exists(data_path):
            self.logger.info(f"Loading test data from: {data_path}")
            # Load from specified file
            data = np.load(data_path, allow_pickle=True)
            X_test = data['features']
            y_test = data['labels']
        else:
            self.logger.info("Loading and preprocessing data from datasets...")
            # Load and preprocess from original datasets
            data_loader = EEGDataLoader()
            preprocessor = EEGPreprocessor(**PREPROCESSING_CONFIG)
            
            # Load datasets
            datasets = data_loader.load_all_datasets()
            
            # Process data (simplified for evaluation)
            all_features = []
            all_labels = []
            
            for dataset_name, data in datasets.items():
                try:
                    processed_data = preprocessor.preprocess_pipeline(
                        data['raw_data'],
                        sampling_rate=data.get('sampling_rate', 256)
                    )
                    
                    features = preprocessor.extract_all_features(processed_data)
                    labels = data_loader._create_intelligent_labels(
                        processed_data, features, num_classes=self.num_classes
                    )
                    
                    all_features.append(features)
                    all_labels.append(labels)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {dataset_name}: {str(e)}")
                    continue
            
            if not all_features:
                raise ValueError("No test data could be loaded")
            
            X_test = np.vstack(all_features)
            y_test = np.hstack(all_labels)
        
        self.logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        self.logger.info(f"Label distribution: {np.bincount(y_test)}")
        
        return X_test, y_test
    
    def evaluate_basic_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic evaluation metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, float]: Basic metrics
        """
        self.logger.info("Calculating basic metrics...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics using model's built-in method
        metrics = self.model.evaluate_model(X_test, y_test)
        
        # Additional metrics
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            metrics[f'accuracy_class_{i}'] = acc
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = np.mean(class_accuracies)
        
        self.logger.info("Basic metrics calculated:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_advanced_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, any]:
        """
        Calculate advanced evaluation metrics including ROC curves and PR curves.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, any]: Advanced metrics and curves
        """
        self.logger.info("Calculating advanced metrics...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        results = {}
        
        # Classification report
        results['classification_report'] = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # ROC curves for multi-class
        if self.num_classes > 2:
            # Binarize labels for multi-class ROC
            y_test_bin = label_binarize(y_test, classes=range(self.num_classes))
            
            results['roc_curves'] = {}
            results['auc_scores'] = {}
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                results['roc_curves'][f'class_{i}'] = {'fpr': fpr, 'tpr': tpr}
                results['auc_scores'][f'class_{i}'] = roc_auc
            
            # Micro-average ROC
            fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
            results['roc_curves']['micro'] = {'fpr': fpr_micro, 'tpr': tpr_micro}
            results['auc_scores']['micro'] = auc(fpr_micro, tpr_micro)
        
        else:
            # Binary classification ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            results['roc_curves'] = {'binary': {'fpr': fpr, 'tpr': tpr}}
            results['auc_scores'] = {'binary': auc(fpr, tpr)}
        
        # Precision-Recall curves
        results['pr_curves'] = {}
        results['ap_scores'] = {}
        
        for i in range(self.num_classes):
            if self.num_classes > 2:
                y_true_class = (y_test == i).astype(int)
                y_scores_class = y_pred_proba[:, i]
            else:
                y_true_class = y_test
                y_scores_class = y_pred_proba[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true_class, y_scores_class)
            ap_score = average_precision_score(y_true_class, y_scores_class)
            
            results['pr_curves'][f'class_{i}'] = {'precision': precision, 'recall': recall}
            results['ap_scores'][f'class_{i}'] = ap_score
        
        self.logger.info("Advanced metrics calculated successfully")
        
        return results
    
    def create_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray, 
                              metrics: Dict, advanced_metrics: Dict) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            X_test: Test features
            y_test: Test labels
            metrics: Basic metrics
            advanced_metrics: Advanced metrics
        """
        self.logger.info("Creating evaluation plots...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 1. Confusion Matrix
        cm_path = os.path.join(self.output_dir, f'confusion_matrix_{self.timestamp}.png')
        self.visualizer.plot_confusion_matrix(
            y_test, y_pred, 
            class_names=self.class_names,
            save_path=cm_path
        )
        
        # 2. ROC Curves
        if 'roc_curves' in advanced_metrics:
            self._plot_roc_curves(advanced_metrics['roc_curves'], 
                                 advanced_metrics['auc_scores'])
        
        # 3. Precision-Recall Curves
        if 'pr_curves' in advanced_metrics:
            self._plot_pr_curves(advanced_metrics['pr_curves'],
                                advanced_metrics['ap_scores'])
        
        # 4. Performance Summary
        self._plot_performance_summary(metrics)
        
        # 5. Prediction Distribution
        self._plot_prediction_distribution(y_test, y_pred, y_pred_proba)
        
        self.logger.info("Evaluation plots created successfully")
    
    def _plot_roc_curves(self, roc_curves: Dict, auc_scores: Dict) -> None:
        """Plot ROC curves."""
        plt.figure(figsize=(10, 8))
        
        if self.num_classes > 2:
            # Multi-class ROC curves
            for i in range(self.num_classes):
                fpr = roc_curves[f'class_{i}']['fpr']
                tpr = roc_curves[f'class_{i}']['tpr']
                auc_score = auc_scores[f'class_{i}']
                
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'{self.class_names[i]} (AUC = {auc_score:.3f})')
            
            # Micro-average
            if 'micro' in roc_curves:
                fpr_micro = roc_curves['micro']['fpr']
                tpr_micro = roc_curves['micro']['tpr']
                auc_micro = auc_scores['micro']
                
                plt.plot(fpr_micro, tpr_micro, linewidth=2, linestyle='--',
                        label=f'Micro-average (AUC = {auc_micro:.3f})')
        else:
            # Binary ROC curve
            fpr = roc_curves['binary']['fpr']
            tpr = roc_curves['binary']['tpr']
            auc_score = auc_scores['binary']
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'ROC Curve (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.output_dir, f'roc_curves_{self.timestamp}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, pr_curves: Dict, ap_scores: Dict) -> None:
        """Plot Precision-Recall curves."""
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            precision = pr_curves[f'class_{i}']['precision']
            recall = pr_curves[f'class_{i}']['recall']
            ap_score = ap_scores[f'class_{i}']
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{self.class_names[i]} (AP = {ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pr_path = os.path.join(self.output_dir, f'pr_curves_{self.timestamp}.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self, metrics: Dict) -> None:
        """Plot performance summary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Main metrics
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [metrics.get(m, 0) for m in main_metrics]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(main_metrics, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('Overall Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Per-class accuracy
        class_accs = [metrics.get(f'accuracy_class_{i}', 0) for i in range(self.num_classes)]
        
        ax2 = axes[0, 1]
        bars = ax2.bar(self.class_names, class_accs, color='skyblue')
        ax2.set_title('Per-Class Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, class_accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Metric comparison radar chart
        ax3 = axes[1, 0]
        self._create_radar_chart(ax3, main_metrics, metric_values)
        
        # Performance vs Threshold (for probability threshold analysis)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Model Performance Summary\n\n' + 
                f"Overall Accuracy: {metrics.get('accuracy', 0):.3f}\n" +
                f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.3f}\n" +
                f"Macro F1-Score: {metrics.get('f1_score', 0):.3f}\n" +
                f"Test Samples: {metrics.get('total_samples', 'N/A')}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(self.output_dir, f'performance_summary_{self.timestamp}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_radar_chart(self, ax, metrics, values):
        """Create a radar chart for metrics."""
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = values + [values[0]]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart')
        ax.grid(True)
    
    def _plot_prediction_distribution(self, y_test: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: np.ndarray) -> None:
        """Plot prediction confidence distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prediction confidence by class
        ax1 = axes[0, 0]
        for i in range(self.num_classes):
            class_probs = y_pred_proba[y_test == i, i]
            ax1.hist(class_probs, bins=20, alpha=0.7, label=self.class_names[i])
        
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Confidence Distribution by True Class')
        ax1.legend()
        
        # Correct vs Incorrect predictions
        ax2 = axes[0, 1]
        correct_mask = (y_test == y_pred)
        correct_probs = np.max(y_pred_proba[correct_mask], axis=1)
        incorrect_probs = np.max(y_pred_proba[~correct_mask], axis=1)
        
        ax2.hist(correct_probs, bins=20, alpha=0.7, label='Correct', color='green')
        ax2.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect', color='red')
        ax2.set_xlabel('Maximum Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Confidence: Correct vs Incorrect')
        ax2.legend()
        
        # Class distribution
        ax3 = axes[1, 0]
        true_counts = np.bincount(y_test)
        pred_counts = np.bincount(y_pred)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        ax3.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax3.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Count')
        ax3.set_title('True vs Predicted Class Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.class_names)
        ax3.legend()
        
        # Prediction uncertainty
        ax4 = axes[1, 1]
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10), axis=1)
        
        for i in range(self.num_classes):
            class_entropy = entropy[y_test == i]
            ax4.boxplot(class_entropy, positions=[i], widths=0.6)
        
        ax4.set_xlabel('True Class')
        ax4.set_ylabel('Prediction Entropy')
        ax4.set_title('Prediction Uncertainty by Class')
        ax4.set_xticks(range(self.num_classes))
        ax4.set_xticklabels(self.class_names)
        
        plt.tight_layout()
        dist_path = os.path.join(self.output_dir, f'prediction_distribution_{self.timestamp}.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, metrics: Dict, advanced_metrics: Dict) -> None:
        """Save evaluation results to files."""
        # Save basic metrics
        basic_results_path = os.path.join(self.output_dir, f'basic_metrics_{self.timestamp}.txt')
        with open(basic_results_path, 'w') as f:
            f.write(f"Model Evaluation Results\n")
            f.write(f"========================\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("Basic Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        # Save detailed classification report
        if 'classification_report' in advanced_metrics:
            report_path = os.path.join(self.output_dir, f'classification_report_{self.timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write("Detailed Classification Report\n")
                f.write("=============================\n\n")
                
                report = advanced_metrics['classification_report']
                
                # Per-class metrics
                for class_name in self.class_names:
                    if class_name in report:
                        class_metrics = report[class_name]
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                        f.write(f"  Support: {class_metrics['support']}\n\n")
                
                # Overall metrics
                if 'macro avg' in report:
                    f.write("Macro Average:\n")
                    macro = report['macro avg']
                    f.write(f"  Precision: {macro['precision']:.4f}\n")
                    f.write(f"  Recall: {macro['recall']:.4f}\n")
                    f.write(f"  F1-Score: {macro['f1-score']:.4f}\n\n")
                
                if 'weighted avg' in report:
                    f.write("Weighted Average:\n")
                    weighted = report['weighted avg']
                    f.write(f"  Precision: {weighted['precision']:.4f}\n")
                    f.write(f"  Recall: {weighted['recall']:.4f}\n")
                    f.write(f"  F1-Score: {weighted['f1-score']:.4f}\n")
        
        # Save AUC scores
        if 'auc_scores' in advanced_metrics:
            auc_path = os.path.join(self.output_dir, f'auc_scores_{self.timestamp}.txt')
            with open(auc_path, 'w') as f:
                f.write("AUC Scores\n")
                f.write("==========\n\n")
                
                for class_key, auc_score in advanced_metrics['auc_scores'].items():
                    f.write(f"{class_key}: {auc_score:.4f}\n")
        
        self.logger.info("Results saved successfully")
    
    def run_complete_evaluation(self, data_path: str = None) -> Dict:
        """
        Run complete model evaluation pipeline.
        
        Args:
            data_path (str): Path to test data (optional)
            
        Returns:
            Dict: Complete evaluation results
        """
        self.logger.info("Starting complete model evaluation...")
        
        try:
            # Load test data
            X_test, y_test = self.load_test_data(data_path)
            
            # Calculate basic metrics
            basic_metrics = self.evaluate_basic_metrics(X_test, y_test)
            basic_metrics['total_samples'] = len(y_test)
            
            # Calculate advanced metrics
            advanced_metrics = self.evaluate_advanced_metrics(X_test, y_test)
            
            # Create plots
            self.create_evaluation_plots(X_test, y_test, basic_metrics, advanced_metrics)
            
            # Save results
            self.save_results(basic_metrics, advanced_metrics)
            
            self.logger.info("Complete evaluation finished successfully")
            
            return {
                'basic_metrics': basic_metrics,
                'advanced_metrics': advanced_metrics,
                'evaluation_dir': self.output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate EEG Stress Detection Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to test data file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Setup directories
    create_directories()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=args.model,
            output_dir=args.output
        )
        
        # Run evaluation
        results = evaluator.run_complete_evaluation(args.data)
        
        # Print summary
        basic_metrics = results['basic_metrics']
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Test Samples: {basic_metrics['total_samples']}")
        print(f"Overall Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {basic_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1-Score: {basic_metrics['f1_score']:.4f}")
        print(f"Results saved to: {results['evaluation_dir']}")
        print("="*50)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
