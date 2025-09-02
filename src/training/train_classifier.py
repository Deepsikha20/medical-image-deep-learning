#!/usr/bin/env python3
"""
Training Script for Medical Image Classification
"""

import os
import argparse
import yaml
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import MedicalImageDataLoader
from models.cnn_classifier import ModelBuilder
from evaluation.metrics import ClassificationMetrics
from utils.callbacks import create_callbacks
from utils.visualization import plot_training_history


class ClassificationTrainer:
    """Medical image classification trainer"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']

        # Create output directories
        self.output_dir = Path(self.config['output_directory'])
        self.model_dir = self.output_dir / 'models'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'

        for dir_path in [self.model_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.data_loader = None
        self.history = None

    def setup_data_loader(self):
        """Setup data loader with configuration"""
        self.data_loader = MedicalImageDataLoader(
            data_dir=self.data_config['data_directory'],
            dataset_name=self.data_config['dataset_name'],
            batch_size=self.training_config['batch_size'],
            image_size=tuple(self.data_config['image_size']),
            num_classes=self.data_config['num_classes'],
            augment=self.training_config.get('augment', True)
        )

        print("Data loader setup complete!")
        dataset_info = self.data_loader.get_dataset_info()
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

    def build_model(self):
        """Build and compile model"""
        print(f"Building {self.model_config['architecture']} model...")

        self.model = ModelBuilder.build_model(
            model_type=self.model_config['architecture'],
            input_shape=(*self.data_config['image_size'], 3),
            num_classes=self.data_config['num_classes'],
            **self.model_config.get('params', {})
        )

        self.model = ModelBuilder.compile_model(
            self.model,
            optimizer=self.training_config['optimizer'],
            learning_rate=self.training_config['learning_rate'],
            loss=self.training_config['loss'],
            metrics=self.training_config.get('metrics', ['accuracy'])
        )

        print(f"Model built with {self.model.count_params():,} parameters")

        # Save model architecture
        with open(self.model_dir / 'model_architecture.json', 'w') as f:
            f.write(self.model.to_json())

    def train_model(self):
        """Train the model"""
        print("Starting model training...")

        # Create data generators
        train_dataset = self.data_loader.create_tf_dataset('train')
        val_dataset = self.data_loader.create_tf_dataset('val')

        # Calculate steps per epoch
        train_steps = len(self.data_loader.splits['train']) // self.training_config['batch_size']
        val_steps = len(self.data_loader.splits['val']) // self.training_config['batch_size']

        # Setup callbacks
        callbacks = create_callbacks(
            model_dir=self.model_dir,
            logs_dir=self.logs_dir,
            patience=self.training_config.get('patience', 10),
            monitor=self.training_config.get('monitor', 'val_loss'),
            save_best_only=True
        )

        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=self.training_config['epochs'],
            validation_data=val_dataset,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        print("Training completed!")

        # Save training history
        history_path = self.logs_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, values in self.history.history.items():
                serializable_history[key] = [float(v) for v in values]
            json.dump(serializable_history, f, indent=2)

    def evaluate_model(self):
        """Evaluate trained model"""
        print("Evaluating model...")

        # Load best model
        best_model_path = self.model_dir / 'best_model.h5'
        if best_model_path.exists():
            self.model = keras.models.load_model(best_model_path)
            print("Loaded best model from training")

        # Create test dataset
        test_dataset = self.data_loader.create_tf_dataset('test')

        # Evaluate on test set
        test_results = self.model.evaluate(test_dataset, verbose=1)

        # Get predictions for detailed analysis
        y_true = []
        y_pred = []

        for batch_images, batch_labels in test_dataset:
            predictions = self.model.predict(batch_images, verbose=0)

            y_true.extend(np.argmax(batch_labels, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))

        # Create evaluation metrics
        metrics_calculator = ClassificationMetrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.data_loader.get_dataset_info()['class_names']
        )

        # Calculate metrics
        metrics = metrics_calculator.calculate_all_metrics()

        # Save evaluation results
        evaluation_results = {
            'test_metrics': dict(zip(self.model.metrics_names, test_results)),
            'detailed_metrics': metrics,
            'model_info': {
                'architecture': self.model_config['architecture'],
                'parameters': int(self.model.count_params()),
                'training_config': self.training_config
            }
        }

        with open(self.logs_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print("Evaluation Results:")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision_macro']:.4f}")
        print(f"Test Recall: {metrics['recall_macro']:.4f}")
        print(f"Test F1-Score: {metrics['f1_macro']:.4f}")

        return evaluation_results

    def generate_visualizations(self):
        """Generate training and evaluation visualizations"""
        print("Generating visualizations...")

        if self.history:
            # Plot training history
            plot_training_history(self.history, save_path=self.plots_dir / 'training_history.png')

        # Create confusion matrix
        if hasattr(self, 'model') and self.model:
            test_dataset = self.data_loader.create_tf_dataset('test')

            y_true = []
            y_pred = []

            for batch_images, batch_labels in test_dataset:
                predictions = self.model.predict(batch_images, verbose=0)
                y_true.extend(np.argmax(batch_labels, axis=1))
                y_pred.extend(np.argmax(predictions, axis=1))

            # Plot confusion matrix
            class_names = self.data_loader.get_dataset_info()['class_names']
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

        print(f"Visualizations saved to: {self.plots_dir}")

    def run_training_pipeline(self):
        """Run complete training pipeline"""
        try:
            print("="*60)
            print("MEDICAL IMAGE CLASSIFICATION TRAINING")
            print("="*60)

            # Setup
            self.setup_data_loader()
            self.build_model()

            # Training
            self.train_model()

            # Evaluation
            evaluation_results = self.evaluate_model()

            # Visualizations
            self.generate_visualizations()

            print("="*60)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)

            return evaluation_results

        except Exception as e:
            print(f"Training pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train medical image classification model')
    parser.add_argument('--config', required=True, help='Training configuration file')
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

    # Run training
    trainer = ClassificationTrainer(args.config)
    results = trainer.run_training_pipeline()

    print("Training completed successfully!")
    print(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
