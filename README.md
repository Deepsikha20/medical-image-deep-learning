# Medical Image Segmentation and Classification using Deep Learning

## 🔬 Project Overview
This project implements deep learning solutions for medical image analysis, specifically focusing on:
- **Classification**: Identifying conditions in medical images (e.g., pneumonia detection in chest X-rays, brain tumor classification)
- **Segmentation**: Precise boundary delineation of anatomical structures or pathological regions

## 🚀 Key Features
- CNN architectures for medical image classification
- U-Net implementation for medical image segmentation  
- Comprehensive preprocessing pipeline
- Multiple evaluation metrics (Accuracy, Precision, Recall, Dice Coefficient, IoU)
- Extensible modular design
- Docker support for reproducible environments

## 📊 Supported Tasks
1. **Chest X-ray Pneumonia Detection** - Binary classification
2. **Brain Tumor MRI Classification** - Multi-class classification  
3. **Medical Image Segmentation** - U-Net for precise boundary detection

## 🛠️ Technologies Used
- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, scikit-image
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, seaborn
- **Evaluation**: scikit-learn metrics

## 📁 Project Structure
```
medical-image-deep-learning/
├── data/                          # Dataset storage
│   ├── raw/                       # Original downloaded datasets
│   ├── processed/                 # Preprocessed data
│   └── augmented/                 # Data augmentation outputs
├── src/                           # Source code
│   ├── data/                      # Data handling modules
│   ├── models/                    # Model architectures
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation utilities
│   └── utils/                     # Helper functions
├── notebooks/                     # Jupyter notebooks for exploration
├── configs/                       # Configuration files
├── requirements.txt               # Dependencies
├── docker/                        # Docker configuration
└── docs/                          # Documentation
```

## 🏃‍♂️ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/medical-image-deep-learning.git
cd medical-image-deep-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets
```bash
# Run dataset download script
python src/data/download_datasets.py
```

### 3. Preprocess Data
```bash
# Preprocess all datasets
python src/data/preprocess_data.py --config configs/preprocess_config.yaml
```

### 4. Train Models
```bash
# Train classification model
python src/training/train_classifier.py --config configs/classification_config.yaml

# Train segmentation model
python src/training/train_segmentation.py --config configs/segmentation_config.yaml
```

### 5. Evaluate Models
```bash
# Evaluate trained models
python src/evaluation/evaluate_model.py --model_path models/trained_model.h5
```

## 📊 Datasets
This project uses publicly available medical imaging datasets:
- **NIH Chest X-Ray Dataset**: 112,120 frontal-view X-ray images from 30,805 unique patients
- **Brain Tumor MRI Dataset**: Multi-class brain tumor classification dataset
- **Medical Segmentation Dataset**: Various anatomical structure segmentation tasks

## 🧠 Model Architectures

### Classification Models
- **Basic CNN**: Custom convolutional neural network
- **ResNet50**: Pre-trained ResNet with transfer learning
- **EfficientNet**: State-of-the-art efficient architecture

### Segmentation Models  
- **U-Net**: Standard encoder-decoder architecture
- **U-Net++**: Enhanced U-Net with nested skip connections
- **Attention U-Net**: U-Net with attention mechanisms

## 📈 Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Segmentation Metrics
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Hausdorff Distance
- Pixel Accuracy

## 🐳 Docker Support
```bash
# Build Docker image
docker build -t medical-image-ml .

# Run container
docker run -v $(pwd):/workspace medical-image-ml
```

## 📚 Documentation
- [Data Preprocessing Guide](docs/preprocessing.md)
- [Model Training Guide](docs/training.md)
- [Evaluation Guide](docs/evaluation.md)
- [API Reference](docs/api.md)

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- NIH Clinical Center for the Chest X-Ray dataset
- Medical imaging research community
- TensorFlow and Keras development teams

## 📞 Contact
Your Name - dasdeepsikha70@gmail.com
Project Link: [https://github.com/Dee/medical-image-deep-learning](https://github.com/Deepsikha20/medical-image-deep-learning)
