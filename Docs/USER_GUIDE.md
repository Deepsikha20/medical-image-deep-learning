# User Guide

This guide covers how to use the Medical Image Deep Learning project for your research and applications.

## Quick Start

### 1. Basic Classification Task

```bash
# Setup data
python src/data/download_datasets.py --sample-only
python src/data/preprocess_data.py --config configs/preprocess_config.yaml

# Train classifier
python src/training/train_classifier.py --config configs/classification_config.yaml

# Evaluate model
python src/evaluation/evaluate_model.py --model-path outputs/classification/models/best_model.h5 --config configs/classification_config.yaml
```

### 2. Segmentation Task

```bash
# Train segmentation model
python src/training/train_segmentation.py --config configs/segmentation_config.yaml

# Evaluate segmentation
python src/evaluation/evaluate_model.py --model-path outputs/segmentation/models/best_model.h5 --config configs/segmentation_config.yaml --task-type segmentation
```

## Configuration

### Dataset Configuration

Edit `configs/dataset_config.yaml`:

```yaml
# Custom dataset
data_directory: "path/to/your/data"
create_sample_data: false

datasets:
  my_dataset:
    name: "My Medical Dataset"
    description: "Custom medical imaging dataset"
```

### Training Configuration

Modify `configs/classification_config.yaml`:

```yaml
data:
  dataset_name: "my_dataset"
  num_classes: 3
  class_names: ["Normal", "Abnormal_Type1", "Abnormal_Type2"]

model:
  architecture: "efficientnetb0"  # or "resnet50", "vgg16", etc.

training:
  batch_size: 16  # Adjust based on GPU memory
  epochs: 100
  learning_rate: 0.0001
```

## Working with Your Own Data

### Data Structure

Organize your data as follows:

```
data/raw/my_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Preprocessing Your Data

```python
from src.data.preprocess_data import MedicalImagePreprocessor

# Initialize preprocessor
preprocessor = MedicalImagePreprocessor('configs/preprocess_config.yaml')

# Process your dataset
preprocessor.process_dataset('my_dataset')
preprocessor.create_data_splits('my_dataset')
```

## Model Architectures

### Available Models

1. **Basic CNN**: Simple convolutional network
2. **ResNet50**: Pre-trained ResNet with transfer learning
3. **EfficientNet**: Efficient and accurate architecture
4. **VGG16**: Classic deep architecture
5. **Attention CNN**: CNN with attention mechanisms

### Custom Models

```python
from src.models.cnn_classifier import ModelBuilder

# Build custom model
model = ModelBuilder.build_model(
    model_type='resnet50',
    input_shape=(224, 224, 3),
    num_classes=4,
    dropout_rate=0.3,
    fine_tune=True
)
```

## Advanced Usage

### Custom Data Loader

```python
from src.data.data_loader import MedicalImageDataLoader

class CustomDataLoader(MedicalImageDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_preprocessing(self, image):
        # Add your custom preprocessing here
        return processed_image
```

### Custom Metrics

```python
from src.evaluation.metrics import ClassificationMetrics

# Add custom evaluation
def custom_metric(y_true, y_pred):
    # Your custom metric calculation
    return metric_value

# Use in evaluation
metrics = ClassificationMetrics(y_true, y_pred, class_names)
custom_result = custom_metric(y_true, y_pred)
```

## Jupyter Notebooks

### Data Exploration

Use `notebooks/01_data_exploration.ipynb` to:
- Analyze dataset statistics
- Visualize class distributions
- Check data quality
- Explore image properties

### Model Training

Use `notebooks/02_model_training.ipynb` to:
- Experiment with different architectures
- Tune hyperparameters
- Monitor training progress
- Visualize results

### Model Evaluation

Use `notebooks/03_model_evaluation.ipynb` to:
- Evaluate model performance
- Analyze prediction errors
- Generate visualizations
- Compare different models

## Best Practices

### Data Preparation

1. **Quality Control**:
   - Remove corrupted images
   - Check image resolutions
   - Verify labels

2. **Data Augmentation**:
   - Use medical-appropriate augmentations
   - Avoid unrealistic transformations
   - Balance augmentation with original data

3. **Class Balance**:
   - Monitor class distributions
   - Use appropriate sampling strategies
   - Consider class weights

### Model Training

1. **Architecture Selection**:
   - Start with pre-trained models
   - Consider domain-specific architectures
   - Balance complexity and performance

2. **Hyperparameter Tuning**:
   - Use validation set for tuning
   - Start with proven configurations
   - Log all experiments

3. **Monitoring**:
   - Use early stopping
   - Monitor multiple metrics
   - Save model checkpoints

### Evaluation

1. **Comprehensive Metrics**:
   - Use appropriate metrics for medical tasks
   - Consider sensitivity/specificity
   - Analyze per-class performance

2. **Validation Strategy**:
   - Use proper train/val/test splits
   - Consider patient-level splits
   - Validate on external datasets

## Troubleshooting

### Training Issues

- **Out of memory**: Reduce batch size or image size
- **Slow training**: Check data loading, use GPU
- **Poor convergence**: Adjust learning rate, check data

### Performance Issues

- **Low accuracy**: Check data quality, try different architectures
- **Overfitting**: Increase regularization, add more data
- **Underfitting**: Increase model complexity, reduce regularization

## Deployment

### Model Export

```python
# Save model
model.save('my_model.h5')

# Export for deployment
tf.saved_model.save(model, 'my_model_savedmodel')
```

### Inference

```python
import tensorflow as tf
from src.data.preprocess_data import MedicalImagePreprocessor

# Load model
model = tf.keras.models.load_model('my_model.h5')

# Preprocess new image
preprocessor = MedicalImagePreprocessor()
processed_image = preprocessor.process_single_image(image_path)

# Make prediction
prediction = model.predict(processed_image)
```

## Contributing

See [CONTRIBUTING.md] for guidelines on:
- Code style
- Testing requirements
- Pull request process
- Issue reporting

## Support

- 📚 Check the [API documentation](API.md)
- 🐛 Report issues on GitHub
- 💬 Join community discussions
- 📧 Contact maintainers
