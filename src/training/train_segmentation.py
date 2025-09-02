#!/usr/bin/env python3
"""
Training Script for Medical Image Segmentation
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
from datetime import datetime

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import SegmentationDataLoader
from models.unet_segmentation import SegmentationModelBuilder
from evaluation.metrics import SegmentationMetrics
from utils.callbacks import create_callbacks
from utils.visualization import plot_training_history, visualize_segmentation_results


class SegmentationTrainer:
    """Medical image segmentation trainer"""

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

    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric for segmentation"""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        """Dice loss function"""
        return 1 - self.dice_coefficient(y_true, y_pred)

    def setup_data_loader(self):
        """Setup segmentation data loader"""
        self.data_loader = SegmentationDataLoader(
            data_dir=self.data_config['data_directory'],
            dataset_name=self.data_config['dataset_name'],
            batch_size=self.training_config['batch_size'],
            image_size=tuple(self.data_config['image_size']),
            num_classes=self.data_config['num_classes'],
            augment=self.training_config.get('augment', True)
        )

        print("Segmentation data loader setup complete!")
        dataset_info = self.data_loader.get_dataset_info()
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

    def build_model(self):
        """Build and compile segmentation model"""
        print(f"Building {self.model_config['architecture']} model...")

        self.model = SegmentationModelBuilder.build_model(
            model_type=self.model_config['architecture'],
            input_shape=(*self.data_config['image_size'], 3),
            num_classes=self.data_config['num_classes'],
            **self.model_config.get('params', {})
        )

        # Custom metrics for segmentation
        metrics = ['accuracy', self.dice_coefficient]

        # Custom loss function
        if self.training_config.get('loss') == 'dice':
            loss = self.dice_loss
        else:
            loss = self.training_config['loss']

        self.model = SegmentationModelBuilder.compile_model(
            self.model,
            optimizer=self.training_config['optimizer'],
            learning_rate=self.training_config['learning_rate'],
            loss=loss,
            metrics=metrics
        )

        print(f"Model built with {self.model.count_params():,} parameters")

        # Save model architecture
        with open(self.model_dir / 'model_architecture.json', 'w') as f:
            f.write(self.model.to_json())

    def train_model(self):
        """Train the segmentation model"""
        print("Starting segmentation model training...")

        # Create data generators (placeholder - would need actual segmentation data)
        # For now, using classification data loader as base
        train_dataset = self.data_loader.create_tf_dataset('train')
        val_dataset = self.data_loader.create_tf_dataset('val')

        # Calculate steps per epoch
        train_steps = len(self.data_loader.splits['train']) // self.training_config['batch_size']
        val_steps = len(self.data_loader.splits['val']) // self.training_config['batch_size']

        # Setup callbacks
        callbacks = create_callbacks(
            model_dir=self.model_dir,
            logs_dir=self.logs_dir,
            patience=self.training_config.get('patience', 15),
            monitor='val_dice_coefficient',  # Monitor Dice coefficient for segmentation
            mode='max',  # Higher Dice is better
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
            serializable_history = {}
            for key, values in self.history.history.items():
                serializable_history[key] = [float(v) for v in values]
            json.dump(serializable_history, f, indent=2)

    def evaluate_model(self):
        """Evaluate trained segmentation model"""
        print("Evaluating segmentation model...")

        # Load best model
        best_model_path = self.model_dir / 'best_model.h5'
        if best_model_path.exists():
            self.model = keras.models.load_model(
                best_model_path,
                custom_objects={
                    'dice_coefficient': self.dice_coefficient,
                    'dice_loss': self.dice_loss
                }
            )
            print("Loaded best model from training")

        # Create test dataset
        test_dataset = self.data_loader.create_tf_dataset('test')

        # Evaluate on test set
        test_results = self.model.evaluate(test_dataset, verbose=1)

        # Detailed evaluation (would need actual masks for real implementation)
        evaluation_results = {
            'test_metrics': dict(zip(self.model.metrics_names, test_results)),
            'model_info': {
                'architecture': self.model_config['architecture'],
                'parameters': int(self.model.count_params()),
                'training_config': self.training_config
            }
        }

        with open(self.logs_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print("Segmentation Evaluation Results:")
        for metric_name, value in zip(self.model.metrics_names, test_results):
            print(f"{metric_name}: {value:.4f}")

        return evaluation_results

    def generate_visualizations(self):
        """Generate training and segmentation visualizations"""
        print("Generating segmentation visualizations...")

        if self.history:
            # Plot training history
            plot_training_history(self.history, save_path=self.plots_dir / 'training_history.png')

        # Generate sample segmentation results (placeholder)
        if hasattr(self, 'model') and self.model:
            test_dataset = self.data_loader.create_tf_dataset('test')

            # Get a batch for visualization
            for batch_images, batch_labels in test_dataset.take(1):
                predictions = self.model.predict(batch_images)

                # Visualize first few samples
                visualize_segmentation_results(
                    images=batch_images[:4],
                    true_masks=batch_labels[:4],
                    pred_masks=predictions[:4],
                    save_path=self.plots_dir / 'segmentation_results.png'
                )
                break

        print(f"Visualizations saved to: {self.plots_dir}")

    def run_training_pipeline(self):
        """Run complete segmentation training pipeline"""
        try:
            print("="*60)
            print("MEDICAL IMAGE SEGMENTATION TRAINING")
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
            print("SEGMENTATION TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)

            return evaluation_results

        except Exception as e:
            print(f"Segmentation training pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
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
    trainer = SegmentationTrainer(args.config)
    results = trainer.run_training_pipeline()

    print("Segmentation training completed successfully!")
    print(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
