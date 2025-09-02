#!/usr/bin/env python3
"""
Visualization Utilities for Medical Image Deep Learning
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def plot_training_history(history, save_path: Optional[Path] = None, 
                         metrics: Optional[List[str]] = None):
    """Plot training history with multiple metrics"""
    if metrics is None:
        # Automatically detect metrics
        metrics = [key for key in history.history.keys() if not key.startswith('val_')]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(5 * ((n_metrics + 1) // 2), 10))

    if n_metrics == 1:
        axes = [axes]
    elif n_metrics <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        train_values = history.history[metric]
        val_metric = f'val_{metric}'
        val_values = history.history.get(val_metric, None)

        epochs = range(1, len(train_values) + 1)

        axes[i].plot(epochs, train_values, 'b-', label=f'Training {metric}')
        if val_values is not None:
            axes[i].plot(epochs, val_values, 'r-', label=f'Validation {metric}')

        axes[i].set_title(f'{metric.capitalize()} Over Epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], 
                         save_path: Optional[Path] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix'):
    """Plot confusion matrix with customization options"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               square=True, linewidths=0.5)

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_data_distribution(data_loader, save_path: Optional[Path] = None):
    """Visualize the distribution of classes in the dataset"""
    dataset_info = data_loader.get_dataset_info()
    class_names = dataset_info['class_names']
    splits = dataset_info['splits']

    # Count samples per class in each split
    split_names = list(splits.keys())
    class_counts = {split: {class_name: 0 for class_name in class_names} for split in split_names}

    # This would need to be implemented based on the actual data structure
    # For now, create sample data
    for split in split_names:
        total_samples = splits[split]
        samples_per_class = total_samples // len(class_names)
        for class_name in class_names:
            class_counts[split][class_name] = samples_per_class

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot of class distribution per split
    x = np.arange(len(class_names))
    width = 0.25

    for i, split in enumerate(split_names):
        counts = [class_counts[split][class_name] for class_name in class_names]
        ax1.bar(x + i * width, counts, width, label=split.capitalize())

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution by Dataset Split')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pie chart of overall class distribution
    total_counts = [sum(class_counts[split][class_name] for split in split_names) 
                   for class_name in class_names]

    ax2.pie(total_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Overall Class Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sample_images(data_loader, split: str = 'train', 
                          num_samples: int = 8, 
                          save_path: Optional[Path] = None):
    """Visualize sample images from the dataset"""
    dataset_info = data_loader.get_dataset_info()
    class_names = dataset_info['class_names']

    # Get sample batch
    dataset = data_loader.create_tf_dataset(split)
    sample_batch = next(iter(dataset.take(1)))
    images, labels = sample_batch

    # Convert labels to class names
    label_indices = np.argmax(labels, axis=1)

    # Create visualization
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten() if rows > 1 else [axes]

    for i in range(min(num_samples, len(images))):
        if i < len(axes):
            axes[i].imshow(images[i])
            axes[i].set_title(f'{class_names[label_indices[i]]}')
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Sample Images from {split.capitalize()} Set', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(model, data_loader, split: str = 'test',
                         num_samples: int = 8, 
                         save_path: Optional[Path] = None):
    """Visualize model predictions on sample images"""
    dataset_info = data_loader.get_dataset_info()
    class_names = dataset_info['class_names']

    # Get sample batch
    dataset = data_loader.create_tf_dataset(split)
    sample_batch = next(iter(dataset.take(1)))
    images, true_labels = sample_batch

    # Get predictions
    predictions = model.predict(images[:num_samples])
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels[:num_samples], axis=1)

    # Create visualization
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten() if rows > 1 else [axes]

    for i in range(min(num_samples, len(images))):
        if i < len(axes):
            axes[i].imshow(images[i])

            true_class = class_names[true_labels[i]]
            pred_class = class_names[pred_labels[i]]
            confidence = predictions[i][pred_labels[i]]

            # Color code: green for correct, red for incorrect
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'

            title = f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}'
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Model Predictions on {split.capitalize()} Set', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_segmentation_results(images: np.ndarray, 
                                 true_masks: np.ndarray,
                                 pred_masks: np.ndarray,
                                 save_path: Optional[Path] = None,
                                 class_names: Optional[List[str]] = None):
    """Visualize segmentation results"""
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')

        # True mask
        true_mask = np.argmax(true_masks[i], axis=-1) if len(true_masks[i].shape) == 3 else true_masks[i]
        axes[i, 1].imshow(true_mask, cmap='tab10')
        axes[i, 1].set_title(f'True Mask {i+1}')
        axes[i, 1].axis('off')

        # Predicted mask
        pred_mask = np.argmax(pred_masks[i], axis=-1) if len(pred_masks[i].shape) == 3 else pred_masks[i]
        axes[i, 2].imshow(pred_mask, cmap='tab10')
        axes[i, 2].set_title(f'Predicted Mask {i+1}')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_feature_maps(model, image: np.ndarray, layer_names: List[str],
                          save_path: Optional[Path] = None):
    """Visualize feature maps from specific layers"""
    # Create a model that outputs the desired layers
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    # Get activations
    activations = activation_model.predict(np.expand_dims(image, axis=0))
    if len(layer_names) == 1:
        activations = [activations]

    # Visualize
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        # Display up to 16 feature maps
        n_cols = 4
        n_display = min(16, n_features)
        n_rows = (n_display + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_rows > 1 else [axes]

        for i in range(n_display):
            if i < len(axes):
                axes[i].imshow(layer_activation[0, :, :, i], cmap='viridis')
                axes[i].set_title(f'Feature {i}')
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_display, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Feature Maps - {layer_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            layer_save_path = save_path.parent / f'{save_path.stem}_{layer_name}{save_path.suffix}'
            plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_model_architecture_plot(model, save_path: Optional[Path] = None):
    """Create a visual representation of the model architecture"""
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=save_path or 'model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=300
        )
        print(f"Model architecture plot saved to: {save_path or 'model_architecture.png'}")
    except ImportError:
        print("Graphviz not available. Install with: pip install graphviz")
    except Exception as e:
        print(f"Error creating model plot: {e}")


def plot_learning_rate_schedule(history, save_path: Optional[Path] = None):
    """Plot learning rate changes during training"""
    if 'lr' in history.history:
        lr_values = history.history['lr']
        epochs = range(1, len(lr_values) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lr_values, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("Learning rate not tracked in history")


def main():
    """Test visualization functions"""
    print("Testing visualization utilities...")

    # Create sample data for testing
    epochs = 20
    sample_history = {
        'loss': np.random.exponential(0.5, epochs)[::-1] + 0.1,
        'accuracy': 1 - np.random.exponential(0.3, epochs)[::-1] * 0.5,
        'val_loss': np.random.exponential(0.6, epochs)[::-1] + 0.15,
        'val_accuracy': 1 - np.random.exponential(0.35, epochs)[::-1] * 0.6,
    }

    class MockHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    history = MockHistory(sample_history)

    # Test plot training history
    print("Testing training history plot...")
    plot_training_history(history)

    print("Visualization testing completed!")


if __name__ == "__main__":
    main()
