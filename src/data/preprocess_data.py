#!/usr/bin/env python3
"""
Data Preprocessing Module
Handles all preprocessing steps for medical images
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import yaml
import json
import matplotlib.pyplot as plt
from skimage import exposure, filters
import albumentations as A


class MedicalImagePreprocessor:
    """Comprehensive medical image preprocessing pipeline"""

    def __init__(self, config_path='configs/preprocess_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_dir = Path(self.config['input_directory'])
        self.output_dir = Path(self.config['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_size = tuple(self.config['target_size'])
        self.normalization_method = self.config.get('normalization_method', 'minmax')

    def load_image(self, image_path, color_mode='RGB'):
        """Load and validate image"""
        try:
            if color_mode == 'RGB':
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # Grayscale
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def resize_image(self, image, target_size=None, maintain_aspect_ratio=True):
        """Resize image with optional aspect ratio preservation"""
        if target_size is None:
            target_size = self.target_size

        if maintain_aspect_ratio:
            # Calculate scaling factor
            h, w = image.shape[:2]
            scale = min(target_size[0]/w, target_size[1]/h)

            # Resize maintaining aspect ratio
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Pad to target size
            if len(image.shape) == 3:
                padded = np.zeros((target_size[1], target_size[0], image.shape[2]), dtype=image.dtype)
            else:
                padded = np.zeros((target_size[1], target_size[0]), dtype=image.dtype)

            # Center the image
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            return padded
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    def normalize_image(self, image, method='minmax'):
        """Normalize image intensity values"""
        image = image.astype(np.float32)

        if method == 'minmax':
            # Min-Max normalization to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif method == 'zscore':
            # Z-score normalization
            image = (image - image.mean()) / (image.std() + 1e-8)
        elif method == 'percentile':
            # Percentile normalization (robust to outliers)
            p1, p99 = np.percentile(image, (1, 99))
            image = np.clip((image - p1) / (p99 - p1), 0, 1)

        return image

    def enhance_contrast(self, image, method='clahe'):
        """Apply contrast enhancement"""
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            if len(image.shape) == 3:
                # Convert to LAB color space for better results
                lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return enhanced.astype(np.float32) / 255.0
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply((image * 255).astype(np.uint8))
                return enhanced.astype(np.float32) / 255.0
        elif method == 'histogram':
            # Standard histogram equalization
            if len(image.shape) == 3:
                enhanced = exposure.equalize_hist(image)
            else:
                enhanced = cv2.equalizeHist((image * 255).astype(np.uint8))
                enhanced = enhanced.astype(np.float32) / 255.0
            return enhanced

        return image

    def denoise_image(self, image, method='gaussian'):
        """Apply denoising"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (3, 3), 0)
        elif method == 'bilateral':
            if len(image.shape) == 3:
                return cv2.bilateralFilter(image, 9, 75, 75)
            else:
                return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur((image * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

        return image

    def create_augmentation_pipeline(self, is_training=True):
        """Create data augmentation pipeline"""
        if is_training:
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
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ])
        else:
            return A.Compose([])  # No augmentation for validation/test

    def process_single_image(self, image_path, output_path, enhance_contrast=True, 
                           denoise=True, augment=False):
        """Process a single image through the complete pipeline"""

        # Load image
        image = self.load_image(image_path)
        if image is None:
            return False

        # Resize image
        image = self.resize_image(image, maintain_aspect_ratio=True)

        # Convert to float32 for processing
        image = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image

        # Apply denoising
        if denoise:
            image = self.denoise_image(image)

        # Enhance contrast
        if enhance_contrast:
            image = self.enhance_contrast(image)

        # Normalize
        image = self.normalize_image(image, self.normalization_method)

        # Apply augmentation if specified
        if augment:
            aug_pipeline = self.create_augmentation_pipeline(is_training=True)
            if len(image.shape) == 3:
                augmented = aug_pipeline(image=(image * 255).astype(np.uint8))
                image = augmented['image'].astype(np.float32) / 255.0

        # Save processed image
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to uint8 for saving
        image_uint8 = (image * 255).astype(np.uint8)

        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)
        else:
            cv2.imwrite(str(output_path), image_uint8)

        return True

    def process_dataset(self, dataset_name):
        """Process entire dataset"""
        print(f"Processing {dataset_name} dataset...")

        input_dataset_dir = self.input_dir / dataset_name
        output_dataset_dir = self.output_dir / dataset_name

        if not input_dataset_dir.exists():
            print(f"Dataset directory not found: {input_dataset_dir}")
            return

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_dataset_dir.rglob(f'*{ext}'))
            image_files.extend(input_dataset_dir.rglob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images to process")

        processed_count = 0
        failed_count = 0

        # Process images with progress bar
        for image_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
            # Maintain directory structure
            relative_path = image_path.relative_to(input_dataset_dir)
            output_path = output_dataset_dir / relative_path

            # Determine if this is training data (for augmentation)
            is_training = 'train' in str(relative_path).lower()

            success = self.process_single_image(
                image_path, 
                output_path,
                augment=is_training and self.config.get('apply_augmentation', False)
            )

            if success:
                processed_count += 1
            else:
                failed_count += 1

        print(f"Processing complete: {processed_count} successful, {failed_count} failed")

        # Save processing statistics
        stats = {
            'dataset': dataset_name,
            'total_images': len(image_files),
            'processed_successfully': processed_count,
            'failed': failed_count,
            'config': self.config
        }

        stats_path = output_dataset_dir / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def create_data_splits(self, dataset_name, test_size=0.2, val_size=0.2):
        """Create train/validation/test splits"""
        dataset_dir = self.output_dir / dataset_name

        # Find all processed images
        image_files = []
        labels = []

        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir() and class_dir.name != 'splits':
                class_name = class_dir.name
                class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

                image_files.extend(class_images)
                labels.extend([class_name] * len(class_images))

        if not image_files:
            print(f"No images found in {dataset_dir}")
            return

        # Create splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_files, labels, test_size=test_size, stratify=labels, random_state=42
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )

        # Save split information
        splits_dir = dataset_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)

        splits_data = {
            'train': [(str(x), y) for x, y in zip(X_train, y_train)],
            'val': [(str(x), y) for x, y in zip(X_val, y_val)],
            'test': [(str(x), y) for x, y in zip(X_test, y_test)]
        }

        with open(splits_dir / 'data_splits.json', 'w') as f:
            json.dump(splits_data, f, indent=2)

        print(f"Data splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    def generate_statistics(self):
        """Generate preprocessing statistics and visualizations"""
        print("Generating preprocessing statistics...")

        stats_dir = self.output_dir / 'statistics'
        stats_dir.mkdir(exist_ok=True)

        # Collect statistics from all datasets
        all_stats = {}

        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != 'statistics':
                stats_file = dataset_dir / 'processing_stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        all_stats[dataset_dir.name] = json.load(f)

        # Save combined statistics
        with open(stats_dir / 'all_preprocessing_stats.json', 'w') as f:
            json.dump(all_stats, f, indent=2)

        print(f"Statistics saved to: {stats_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess medical imaging datasets')
    parser.add_argument('--config', default='configs/preprocess_config.yaml',
                       help='Preprocessing configuration file')
    parser.add_argument('--dataset', help='Specific dataset to process')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')

    args = parser.parse_args()

    preprocessor = MedicalImagePreprocessor(args.config)

    if args.dataset:
        preprocessor.process_dataset(args.dataset)
        if args.create_splits:
            preprocessor.create_data_splits(args.dataset)
    else:
        # Process all datasets
        for dataset_name in ['chest_xray', 'brain_tumor']:
            if (preprocessor.input_dir / dataset_name).exists():
                preprocessor.process_dataset(dataset_name)
                if args.create_splits:
                    preprocessor.create_data_splits(dataset_name)

    preprocessor.generate_statistics()


if __name__ == "__main__":
    main()
