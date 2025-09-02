#!/usr/bin/env python3
"""
Medical Image Data Loading Module
Handles loading and batching of preprocessed medical images
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
import random
from typing import Tuple, List, Dict, Optional
import albumentations as A


class MedicalImageDataLoader:
    """Data loader for medical imaging datasets"""

    def __init__(self, 
                 data_dir: str,
                 dataset_name: str,
                 batch_size: int = 32,
                 image_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 2,
                 augment: bool = True):

        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment

        self.dataset_dir = self.data_dir / dataset_name
        self.splits_file = self.dataset_dir / 'splits' / 'data_splits.json'

        # Load data splits
        self._load_splits()

        # Create augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()

    def _load_splits(self):
        """Load train/val/test splits from JSON file"""
        if self.splits_file.exists():
            with open(self.splits_file, 'r') as f:
                self.splits = json.load(f)
        else:
            print(f"Splits file not found: {self.splits_file}")
            print("Creating splits from directory structure...")
            self._create_splits_from_structure()

    def _create_splits_from_structure(self):
        """Create splits from directory structure if splits file doesn't exist"""
        # This is a fallback method if splits file doesn't exist
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        splits = {'train': [], 'val': [], 'test': []}

        # Look for standard directory structures
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_file in class_dir.iterdir():
                            if img_file.suffix.lower() in image_extensions:
                                splits[split].append((str(img_file), class_name))

        # If no standard structure, create splits from all images
        if not any(splits.values()):
            all_images = []
            for class_dir in self.dataset_dir.iterdir():
                if class_dir.is_dir() and class_dir.name != 'splits':
                    class_name = class_dir.name
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            all_images.append((str(img_file), class_name))

            # Shuffle and split
            random.shuffle(all_images)
            total = len(all_images)
            train_end = int(0.7 * total)
            val_end = int(0.85 * total)

            splits['train'] = all_images[:train_end]
            splits['val'] = all_images[train_end:val_end]
            splits['test'] = all_images[val_end:]

        self.splits = splits

    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline using Albumentations"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to target size
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((*self.image_size, 3), dtype=np.float32)

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image"""
        # Convert to uint8 for augmentation
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image_uint8)

        # Convert back to float32
        return augmented['image'].astype(np.float32) / 255.0

    def create_label_encoder(self) -> Dict[str, int]:
        """Create mapping from class names to integers"""
        all_labels = set()
        for split_data in self.splits.values():
            for _, label in split_data:
                all_labels.add(label)

        return {label: idx for idx, label in enumerate(sorted(all_labels))}

    def data_generator(self, split: str = 'train'):
        """Generator function for creating batches"""
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not found in data")

        split_data = self.splits[split]
        label_encoder = self.create_label_encoder()

        # Shuffle data for training
        if split == 'train':
            random.shuffle(split_data)

        batch_images = []
        batch_labels = []

        for image_path, label in split_data:
            # Load image
            image = self.load_image(image_path)

            # Apply augmentation for training
            if split == 'train' and self.augment:
                image = self.augment_image(image)

            # Encode label
            label_encoded = label_encoder[label]

            batch_images.append(image)
            batch_labels.append(label_encoded)

            # Yield batch when full
            if len(batch_images) == self.batch_size:
                yield (
                    np.array(batch_images),
                    tf.keras.utils.to_categorical(batch_labels, self.num_classes)
                )
                batch_images = []
                batch_labels = []

        # Yield remaining images if any
        if batch_images:
            yield (
                np.array(batch_images),
                tf.keras.utils.to_categorical(batch_labels, self.num_classes)
            )

    def create_tf_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """Create TensorFlow dataset"""
        def generator():
            return self.data_generator(split)

        # Determine output signature
        output_signature = (
            tf.TensorSpec(shape=(None, *self.image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        # Add prefetching for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        label_encoder = self.create_label_encoder()

        info = {
            'dataset_name': self.dataset_name,
            'num_classes': len(label_encoder),
            'class_names': list(label_encoder.keys()),
            'splits': {
                split: len(data) for split, data in self.splits.items()
            },
            'image_size': self.image_size,
            'batch_size': self.batch_size
        }

        return info

    def visualize_batch(self, split: str = 'train', num_images: int = 8):
        """Visualize a batch of images"""
        import matplotlib.pyplot as plt

        # Get a batch
        batch_gen = self.data_generator(split)
        images, labels = next(batch_gen)

        # Convert one-hot labels back to class names
        label_encoder = self.create_label_encoder()
        class_names = list(label_encoder.keys())
        label_indices = np.argmax(labels, axis=1)

        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i in range(min(num_images, len(images))):
            axes[i].imshow(images[i])
            axes[i].set_title(f'{class_names[label_indices[i]]}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


class SegmentationDataLoader(MedicalImageDataLoader):
    """Data loader specifically for segmentation tasks"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_image_mask_pair(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and corresponding segmentation mask"""
        # Load image
        image = self.load_image(image_path)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Create empty mask as fallback
            mask = np.zeros(self.image_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Convert mask to categorical
        mask = tf.keras.utils.to_categorical(mask, self.num_classes)

        return image, mask

    def augment_image_mask_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to both image and mask"""
        # Convert to uint8 for augmentation
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Apply same transform to both image and mask
        augmented = self.augmentation_pipeline(image=image_uint8, mask=mask_uint8)

        # Convert back
        aug_image = augmented['image'].astype(np.float32) / 255.0
        aug_mask = augmented['mask'].astype(np.float32) / 255.0

        return aug_image, aug_mask


def main():
    """Test data loader functionality"""
    # Example usage
    data_loader = MedicalImageDataLoader(
        data_dir='data/processed',
        dataset_name='chest_xray',
        batch_size=16,
        image_size=(224, 224),
        num_classes=2
    )

    # Print dataset info
    info = data_loader.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test generator
    print("
Testing data generator...")
    train_gen = data_loader.data_generator('train')
    batch_images, batch_labels = next(train_gen)
    print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")

    # Visualize batch (uncomment to show)
    # data_loader.visualize_batch('train')


if __name__ == "__main__":
    main()
