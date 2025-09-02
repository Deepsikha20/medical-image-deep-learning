#!/usr/bin/env python3
"""
Model Evaluation Script for Medical Image Analysis
"""

import os
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import MedicalImageDataLoader, SegmentationDataLoader
from evaluation.metrics import ClassificationMetrics, SegmentationMetrics
from utils.visualization import plot_confusion_matrix, visualize_predictions


class ModelEvaluator:
    """Comprehensive model evaluation for medical images"""

    def __init__(self, model_path: str, config_path: str, task_type: str = 'classification'):
        self.model_path = Path(model_path)
        self.config_path = config_path
        self.task_type = task_type.lower()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.evaluation_config = self.config.get('evaluation', {})

        # Create output directory
        self.output_dir = Path(self.config.get('output_directory', 'outputs/evaluation'))
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'

        for dir_path in [self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.data_loader = None

    def load_model(self):
        """Load trained model"""
        print(f"Loading model from: {self.model_path}")

        # Custom objects for loading models with custom metrics/losses
        custom_objects = {}

        if self.task_type == 'segmentation':
            def dice_coefficient(y_true, y_pred, smooth=1e-6):
                y_true_f = tf.reshape(y_true, [-1])
                y_pred_f = tf.reshape(y_pred, [-1])
                intersection = tf.reduce_sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

            def dice_loss(y_true, y_pred):
                return 1 - dice_coefficient(y_true, y_pred)

            custom_objects.update({
                'dice_coefficient': dice_coefficient,
                'dice_loss': dice_loss
            })

        try:
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
            print(f"Model loaded successfully. Parameters: {self.model.count_params():,}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def setup_data_loader(self):
        """Setup data loader based on task type"""
        if self.task_type == 'classification':
            self.data_loader = MedicalImageDataLoader(
                data_dir=self.data_config['data_directory'],
                dataset_name=self.data_config['dataset_name'],
                batch_size=self.evaluation_config.get('batch_size', 32),
                image_size=tuple(self.data_config['image_size']),
                num_classes=self.data_config['num_classes'],
                augment=False  # No augmentation for evaluation
            )
        elif self.task_type == 'segmentation':
            self.data_loader = SegmentationDataLoader(
                data_dir=self.data_config['data_directory'],
                dataset_name=self.data_config['dataset_name'],
                batch_size=self.evaluation_config.get('batch_size', 16),
                image_size=tuple(self.data_config['image_size']),
                num_classes=self.data_config['num_classes'],
                augment=False
            )

        print("Data loader setup complete!")
        dataset_info = self.data_loader.get_dataset_info()
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

    def evaluate_classification_model(self):
        """Evaluate classification model"""
        print("Evaluating classification model...")

        # Create test dataset
        test_dataset = self.data_loader.create_tf_dataset('test')

        # Get model predictions and true labels
        y_true = []
        y_pred = []
        y_prob = []

        print("Generating predictions...")
        for batch_images, batch_labels in test_dataset:
            predictions = self.model.predict(batch_images, verbose=0)

            y_true.extend(np.argmax(batch_labels, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
            y_prob.extend(predictions)

        y_prob = np.array(y_prob)

        # Calculate comprehensive metrics
        class_names = self.data_loader.get_dataset_info()['class_names']

        metrics_calculator = ClassificationMetrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names
        )

        all_metrics = metrics_calculator.calculate_all_metrics()

        # Print results
        print("" + "="*60)
        print("CLASSIFICATION EVALUATION RESULTS")
        print("="*60)

        basic_metrics = all_metrics['basic_metrics']
        print("Overall Metrics:")
        print(f"  Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"  Precision (Macro): {basic_metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro): {basic_metrics['recall_macro']:.4f}")
        print(f"  F1-Score (Macro): {basic_metrics['f1_macro']:.4f}")

        if all_metrics['auc_metrics']['auc_macro'] is not None:
            print(f"  AUC (Macro): {all_metrics['auc_metrics']['auc_macro']:.4f}")

        print("Per-Class Metrics:")
        for class_name, class_metrics in all_metrics['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1-Score: {class_metrics['f1_score']:.4f}")

        # Generate visualizations
        self._generate_classification_visualizations(y_true, y_pred, y_prob, class_names, all_metrics)

        # Save detailed report
        with open(self.reports_dir / 'classification_evaluation.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)

        # Save classification report
        with open(self.reports_dir / 'classification_report.txt', 'w') as f:
            f.write(all_metrics['classification_report'])

        return all_metrics

    def evaluate_segmentation_model(self):
        """Evaluate segmentation model"""
        print("Evaluating segmentation model...")

        # Create test dataset
        test_dataset = self.data_loader.create_tf_dataset('test')

        # Evaluate model
        test_results = self.model.evaluate(test_dataset, verbose=1)
        basic_results = dict(zip(self.model.metrics_names, test_results))

        # Get predictions for detailed analysis
        all_true_masks = []
        all_pred_masks = []

        print("Generating segmentation predictions...")
        for batch_images, batch_masks in test_dataset:
            predictions = self.model.predict(batch_images, verbose=0)

            # Convert predictions to class indices
            pred_masks = np.argmax(predictions, axis=-1)
            true_masks = np.argmax(batch_masks, axis=-1) if len(batch_masks.shape) == 4 else batch_masks

            all_true_masks.extend(true_masks)
            all_pred_masks.extend(pred_masks)

        # Calculate segmentation metrics
        all_true_masks = np.array(all_true_masks)
        all_pred_masks = np.array(all_pred_masks)

        class_names = self.data_loader.get_dataset_info().get('class_names', 
                                                             [f'Class_{i}' for i in range(self.data_config['num_classes'])])

        seg_metrics = SegmentationMetrics(
            y_true=all_true_masks,
            y_pred=all_pred_masks,
            num_classes=self.data_config['num_classes'],
            class_names=class_names
        )

        detailed_metrics = seg_metrics.calculate_all_metrics()

        # Combine results
        all_metrics = {
            'basic_results': basic_results,
            'detailed_metrics': detailed_metrics
        }

        # Print results
        print("" + "="*60)
        print("SEGMENTATION EVALUATION RESULTS")
        print("="*60)

        print("Model Metrics:")
        for metric_name, value in basic_results.items():
            print(f"  {metric_name}: {value:.4f}")

        overall_metrics = detailed_metrics['overall_metrics']
        print("Overall Segmentation Metrics:")
        print(f"  Mean Dice Coefficient: {overall_metrics['mean_dice']:.4f}")
        print(f"  Mean Jaccard Index: {overall_metrics['mean_jaccard']:.4f}")
        print(f"  Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")

        print("Per-Class Segmentation Metrics:")
        for class_name, class_metrics in detailed_metrics['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Dice Coefficient: {class_metrics['dice_coefficient']:.4f}")
            print(f"    Jaccard Index: {class_metrics['jaccard_index']:.4f}")

        # Generate visualizations
        self._generate_segmentation_visualizations(all_true_masks[:4], all_pred_masks[:4], class_names)

        # Save results
        with open(self.reports_dir / 'segmentation_evaluation.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics

    def _generate_classification_visualizations(self, y_true, y_pred, y_prob, class_names, metrics):
        """Generate classification visualizations"""
        print("Generating classification visualizations...")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ROC Curves (if probabilities available)
        if y_prob is not None and len(class_names) >= 2:
            from evaluation.metrics import MedicalImageMetrics
            MedicalImageMetrics.plot_roc_curve(
                np.array(y_true), y_prob, class_names,
                save_path=self.plots_dir / 'roc_curves.png'
            )

        # Metrics summary plot
        basic_metrics = metrics['basic_metrics']
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_values = [basic_metrics[name] for name in metric_names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
        plt.ylim(0, 1)
        plt.title('Classification Metrics Summary')
        plt.ylabel('Score')

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_segmentation_visualizations(self, true_masks, pred_masks, class_names):
        """Generate segmentation visualizations"""
        print("Generating segmentation visualizations...")

        # Sample segmentation results
        n_samples = min(4, len(true_masks))
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))

        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_samples):
            # True mask
            axes[0, i].imshow(true_masks[i], cmap='tab10', vmin=0, vmax=len(class_names)-1)
            axes[0, i].set_title(f'True Mask {i+1}')
            axes[0, i].axis('off')

            # Predicted mask
            axes[1, i].imshow(pred_masks[i], cmap='tab10', vmin=0, vmax=len(class_names)-1)
            axes[1, i].set_title(f'Predicted Mask {i+1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'segmentation_samples.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        try:
            print("="*60)
            print(f"MEDICAL IMAGE {self.task_type.upper()} EVALUATION")
            print("="*60)

            # Setup
            self.load_model()
            self.setup_data_loader()

            # Evaluation
            if self.task_type == 'classification':
                results = self.evaluate_classification_model()
            elif self.task_type == 'segmentation':
                results = self.evaluate_segmentation_model()
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

            print("" + "="*60)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Results saved to: {self.output_dir}")

            return results

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Evaluate medical image analysis model')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--config', required=True, help='Evaluation configuration file')
    parser.add_argument('--task-type', choices=['classification', 'segmentation'], 
                       default='classification', help='Type of task to evaluate')
    parser.add_argument('--gpu', type=int, help='GPU device to use')

    args = parser.parse_args()

    # Setup GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Configure TensorFlow GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")

    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.config, args.task_type)
    results = evaluator.run_evaluation()

    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
