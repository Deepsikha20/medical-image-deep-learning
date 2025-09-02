#!/usr/bin/env python3
"""
Dataset Download Module
Handles downloading and organizing medical imaging datasets
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml


class DatasetDownloader:
    """Download and organize medical imaging datasets"""

    def __init__(self, config_path='configs/dataset_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(self.config['data_directory'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filepath):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as file, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)

    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)  # Clean up zip file

    def download_chest_xray_dataset(self):
        """Download NIH Chest X-Ray dataset"""
        print("Downloading NIH Chest X-Ray Dataset...")

        # Note: This is a placeholder - actual NIH dataset requires special access
        # Users would need to manually download from official sources
        dataset_dir = self.data_dir / 'chest_xray'
        dataset_dir.mkdir(exist_ok=True)

        # Create sample directory structure
        for split in ['train', 'val', 'test']:
            for category in ['NORMAL', 'PNEUMONIA']:
                (dataset_dir / split / category).mkdir(parents=True, exist_ok=True)

        print(f"Dataset directory created at: {dataset_dir}")
        print("Please manually download NIH Chest X-Ray dataset from official source")

    def download_brain_tumor_dataset(self):
        """Download Brain Tumor MRI dataset"""
        print("Downloading Brain Tumor MRI Dataset...")

        # Kaggle dataset would require API key setup
        dataset_dir = self.data_dir / 'brain_tumor'
        dataset_dir.mkdir(exist_ok=True)

        # Create sample directory structure
        categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        for split in ['Training', 'Testing']:
            for category in categories:
                (dataset_dir / split / category).mkdir(parents=True, exist_ok=True)

        print(f"Dataset directory created at: {dataset_dir}")
        print("Please manually download Brain Tumor dataset from Kaggle")

    def create_sample_data(self):
        """Create sample data for testing"""
        import numpy as np
        from PIL import Image

        print("Creating sample data for testing...")

        # Create sample chest X-ray data
        chest_dir = self.data_dir / 'chest_xray'
        for split in ['train', 'val']:
            for category in ['NORMAL', 'PNEUMONIA']:
                category_dir = chest_dir / split / category
                category_dir.mkdir(parents=True, exist_ok=True)

                # Create 5 sample images per category
                for i in range(5):
                    # Generate random grayscale image
                    img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
                    img = Image.fromarray(img_array, 'L')
                    img.save(category_dir / f'sample_{i:03d}.jpeg')

        # Create sample brain tumor data
        brain_dir = self.data_dir / 'brain_tumor'
        categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        for split in ['Training', 'Testing']:
            for category in categories:
                category_dir = brain_dir / split / category
                category_dir.mkdir(parents=True, exist_ok=True)

                # Create 3 sample images per category
                for i in range(3):
                    # Generate random RGB image
                    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array, 'RGB')
                    img.save(category_dir / f'sample_{i:03d}.jpg')

        print("Sample data created successfully!")

    def download_all(self):
        """Download all configured datasets"""
        if self.config.get('download_chest_xray', True):
            self.download_chest_xray_dataset()

        if self.config.get('download_brain_tumor', True):
            self.download_brain_tumor_dataset()

        if self.config.get('create_sample_data', False):
            self.create_sample_data()


def main():
    parser = argparse.ArgumentParser(description='Download medical imaging datasets')
    parser.add_argument('--config', default='configs/dataset_config.yaml',
                       help='Dataset configuration file')
    parser.add_argument('--sample-only', action='store_true',
                       help='Create sample data only')

    args = parser.parse_args()

    downloader = DatasetDownloader(args.config)

    if args.sample_only:
        downloader.create_sample_data()
    else:
        downloader.download_all()


if __name__ == "__main__":
    main()
