#!/usr/bin/env python3
"""
Training Callbacks for Medical Image Deep Learning
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json


def create_callbacks(model_dir: Path, 
                    logs_dir: Path,
                    patience: int = 10,
                    monitor: str = 'val_loss',
                    mode: str = 'min',
                    save_best_only: bool = True,
                    restore_best_weights: bool = True,
                    reduce_lr_patience: int = 5,
                    reduce_lr_factor: float = 0.5) -> list:
    """Create standard callbacks for medical image training"""

    callbacks = []

    # Model Checkpoint - Save best model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / 'best_model.h5'),
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Early Stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=1
    )
    callbacks.append(early_stopping_callback)

    # Reduce Learning Rate on Plateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        mode=mode,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)

    # TensorBoard Logging
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(logs_dir / 'tensorboard'),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)

    # CSV Logger
    csv_logger_callback = keras.callbacks.CSVLogger(
        filename=str(logs_dir / 'training_log.csv'),
        separator=',',
        append=False
    )
    callbacks.append(csv_logger_callback)

    # Custom callback for saving epoch-wise model
    save_epoch_callback = SaveEpochModelCallback(model_dir)
    callbacks.append(save_epoch_callback)

    return callbacks


class SaveEpochModelCallback(keras.callbacks.Callback):
    """Custom callback to save model at specific epochs"""

    def __init__(self, model_dir: Path, save_frequency: int = 10):
        super().__init__()
        self.model_dir = model_dir
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            model_path = self.model_dir / f'model_epoch_{epoch+1:03d}.h5'
            self.model.save(str(model_path))
            print(f"Model saved at epoch {epoch+1}: {model_path}")


class LearningRateSchedulerCallback(keras.callbacks.Callback):
    """Custom learning rate scheduler for medical images"""

    def __init__(self, schedule_type: str = 'cosine', initial_lr: float = 0.001,
                 min_lr: float = 1e-6, warmup_epochs: int = 5):
        super().__init__()
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warm-up phase
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            if self.schedule_type == 'cosine':
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.params['epochs'] - self.warmup_epochs)
                lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
            elif self.schedule_type == 'step':
                # Step decay
                decay_epochs = [30, 60, 90]
                lr = self.initial_lr
                for decay_epoch in decay_epochs:
                    if epoch >= decay_epoch:
                        lr *= 0.1
            else:
                lr = self.initial_lr

        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"Epoch {epoch+1}: Learning rate = {lr:.2e}")


class MetricsCallback(keras.callbacks.Callback):
    """Custom callback to track and save additional metrics"""

    def __init__(self, logs_dir: Path, validation_data=None):
        super().__init__()
        self.logs_dir = logs_dir
        self.validation_data = validation_data
        self.metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Record metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }

        # Additional custom metrics can be calculated here
        if self.validation_data is not None:
            # Custom validation metrics
            pass

        self.metrics_history.append(epoch_metrics)

        # Save metrics history
        with open(self.logs_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class GradientNormCallback(keras.callbacks.Callback):
    """Callback to monitor gradient norms during training"""

    def __init__(self, logs_dir: Path, log_frequency: int = 1):
        super().__init__()
        self.logs_dir = logs_dir
        self.log_frequency = log_frequency
        self.gradient_norms = []

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            # Calculate gradient norms
            gradients = []
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    grad = tf.gradients(self.model.total_loss, layer.kernel)[0]
                    if grad is not None:
                        gradients.append(tf.norm(grad))

            if gradients:
                grad_norm = tf.reduce_mean(gradients)
                self.gradient_norms.append({
                    'batch': batch,
                    'gradient_norm': float(grad_norm.numpy())
                })

    def on_epoch_end(self, epoch, logs=None):
        # Save gradient norms
        if self.gradient_norms:
            with open(self.logs_dir / f'gradient_norms_epoch_{epoch+1}.json', 'w') as f:
                json.dump(self.gradient_norms, f, indent=2)
            self.gradient_norms = []


class ValidationVisualizationCallback(keras.callbacks.Callback):
    """Callback to visualize validation predictions during training"""

    def __init__(self, validation_data, logs_dir: Path, 
                 class_names: list, num_samples: int = 4,
                 save_frequency: int = 5):
        super().__init__()
        self.validation_data = validation_data
        self.logs_dir = logs_dir
        self.class_names = class_names
        self.num_samples = num_samples
        self.save_frequency = save_frequency

        # Create visualization directory
        self.viz_dir = logs_dir / 'validation_visualizations'
        self.viz_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            self._visualize_predictions(epoch + 1)

    def _visualize_predictions(self, epoch):
        """Create visualization of validation predictions"""
        # Get a batch of validation data
        val_batch = next(iter(self.validation_data.take(1)))
        images, true_labels = val_batch

        # Get predictions
        predictions = self.model.predict(images[:self.num_samples])
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(true_labels[:self.num_samples], axis=1)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()

        for i in range(min(self.num_samples, len(axes))):
            axes[i].imshow(images[i])

            true_class = self.class_names[true_labels[i]]
            pred_class = self.class_names[pred_labels[i]]
            confidence = predictions[i][pred_labels[i]]

            title = f'True: {true_class}\nPred: {pred_class} ({confidence:.3f})'
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'

            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'predictions_epoch_{epoch:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


class MemoryUsageCallback(keras.callbacks.Callback):
    """Callback to monitor memory usage during training"""

    def __init__(self, logs_dir: Path):
        super().__init__()
        self.logs_dir = logs_dir
        self.memory_usage = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            import psutil
            import GPUtil

            # CPU memory
            cpu_memory = psutil.virtual_memory()

            # GPU memory (if available)
            gpu_memory = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = {
                        'used': gpus[0].memoryUsed,
                        'total': gpus[0].memoryTotal,
                        'utilization': gpus[0].memoryUtil * 100
                    }
            except:
                pass

            memory_info = {
                'epoch': epoch + 1,
                'cpu_memory_percent': cpu_memory.percent,
                'cpu_memory_used_gb': cpu_memory.used / (1024**3),
                'cpu_memory_total_gb': cpu_memory.total / (1024**3),
                'gpu_memory': gpu_memory
            }

            self.memory_usage.append(memory_info)

            # Save memory usage
            with open(self.logs_dir / 'memory_usage.json', 'w') as f:
                json.dump(self.memory_usage, f, indent=2)

        except ImportError:
            pass  # Skip if psutil/GPUtil not available


def main():
    """Test callback functionality"""
    from pathlib import Path

    # Create test directories
    model_dir = Path('test_outputs/models')
    logs_dir = Path('test_outputs/logs')
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create test callbacks
    callbacks = create_callbacks(
        model_dir=model_dir,
        logs_dir=logs_dir,
        patience=5,
        monitor='val_loss'
    )

    print(f"Created {len(callbacks)} callbacks:")
    for i, callback in enumerate(callbacks):
        print(f"  {i+1}. {type(callback).__name__}")

    print("Callback testing completed!")


if __name__ == "__main__":
    main()
