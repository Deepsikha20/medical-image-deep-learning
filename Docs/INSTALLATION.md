# Installation Guide

This guide will help you set up the Medical Image Deep Learning project on your system.

## System Requirements

### Hardware
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.2+ (if using GPU)
- **Git**: For cloning the repository

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/medical-image-deep-learning.git
   cd medical-image-deep-learning
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/setup_environment.sh
   ./scripts/setup_environment.sh
   ```

3. **Activate the environment:**
   ```bash
   source venv/bin/activate
   ```

### Method 2: Manual Setup

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/your-username/medical-image-deep-learning.git
   cd medical-image-deep-learning
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create directories:**
   ```bash
   mkdir -p data/{raw,processed,augmented}
   mkdir -p outputs/{models,logs,plots,reports}
   ```

### Method 3: Docker Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/medical-image-deep-learning.git
   cd medical-image-deep-learning
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access Jupyter Lab:**
   - Open browser to `http://localhost:8888`
   - TensorBoard available at `http://localhost:6006`

## GPU Setup (Optional)

### NVIDIA GPU with CUDA

1. **Install NVIDIA drivers:**
   - Download from [NVIDIA website](https://www.nvidia.com/drivers/)

2. **Install CUDA Toolkit:**
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-cuda-toolkit

   # Or download from NVIDIA website
   ```

3. **Verify installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

4. **Test TensorFlow GPU:**
   ```python
   import tensorflow as tf
   print("GPUs:", tf.config.list_physical_devices('GPU'))
   ```

## Verification

Test your installation:

```bash
# Activate environment
source venv/bin/activate

# Test imports
python -c "
import tensorflow as tf
import cv2
import sklearn
import matplotlib
print('✅ All imports successful')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
"
```

## Next Steps

1. **Download datasets:**
   ```bash
   python src/data/download_datasets.py
   ```

2. **Explore the notebooks:**
   ```bash
   jupyter lab notebooks/
   ```

3. **Run a quick training test:**
   ```bash
   python src/data/download_datasets.py --sample-only
   python src/data/preprocess_data.py
   ./scripts/run_training.sh
   ```

## Troubleshooting

### Common Issues

1. **Python version error:**
   - Ensure Python 3.8+ is installed
   - Use `python3` instead of `python` if needed

2. **Permission denied:**
   ```bash
   chmod +x scripts/*.sh
   ```

3. **CUDA/GPU issues:**
   - Verify NVIDIA drivers and CUDA installation
   - Check TensorFlow GPU compatibility

4. **Memory errors:**
   - Reduce batch size in config files
   - Use smaller image sizes
   - Enable mixed precision training

5. **Package conflicts:**
   ```bash
   pip install --upgrade --force-reinstall tensorflow
   ```

### Getting Help

- Check the [FAQ](FAQ.md)
- Open an issue on GitHub
- Review the [documentation](API.md)

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```
