# 🔍 VerifyVision-Pro 🖼️

<div align="center">
  
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**深度学习驱动的图像伪造检测系统**  
**A Deep Learning-Powered Image Forgery Detection System**

[English](#english-documentation) | [中文](#chinese-documentation)

</div>

---

<a name="english-documentation"></a>
# English Documentation 🌍

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [Technical Implementation](#technical-implementation)
- [Advanced Usage](#advanced-usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Notes & Best Practices](#notes)
- [Contributing](#contributing)
- [License](#license)

## 🔭 Overview <a name="overview"></a>

VerifyVision-Pro is a comprehensive deep learning-based system designed to detect image forgeries with high accuracy. The system integrates robust data processing pipelines, state-of-the-art deep learning models, and an intuitive web interface for real-time detection.

### 🌟 Key Features

- **Multi-model Support**: Implements various architectures (EfficientNet, ResNet, Xception, CNN)
- **Comprehensive Pipeline**: Complete workflow from data preparation to deployment
- **User-friendly Interface**: Web-based UI for easy interaction with the system
- **Detailed Analytics**: Provides confidence scores and visualization of results
- **Optimized Performance**: Supports both CPU and GPU inference

## 📁 Project Structure <a name="project-structure"></a>

```
VerifyVision-Pro/
│
├── data/                      # Data directory
│   ├── real/                  # Real images
│   ├── fake/                  # Forged images
│   └── processed/             # Preprocessed images
│
├── models/                    # Model directory (gitignored)
│   └── saved/                 # Saved model weights
│
├── src/                       # Source code
│   ├── data_utils/            # Data processing utilities
│   │   ├── dataset.py         # Dataset class
│   │   └── data_processor.py  # Data preprocessing tools
│   │
│   ├── models/                # Model definitions
│   │   └── models.py          # Deep learning model implementations
│   │
│   ├── training/              # Training related
│   │   ├── train.py           # Training scripts
│   │   └── evaluate.py        # Evaluation scripts
│   │
│   └── web/                   # Web application
│       └── app.py             # Flask application
│
├── static/                    # Static resources
│   ├── css/                   # CSS styles
│   │   └── style.css          # Custom styles
│   │
│   ├── js/                    # JavaScript
│   │   └── main.js            # Main JS file
│   │
│   └── uploads/               # User uploaded images
│
├── templates/                 # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   ├── result.html            # Results page
│   └── about.html             # About page
│
├── generate_test_images.py    # Test image generation script
├── main.py                    # Project main entry program
├── requirements.txt           # Project dependencies
└── README.md                  # Project description
```

## 💻 System Requirements <a name="system-requirements"></a>

### Minimum Requirements

- **Python**: 3.7+
- **PyTorch**: 2.0+
- **RAM**: 4GB (CPU only), 8GB (with GPU)
- **Storage**: 1GB for code and basic datasets
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+ with CUDA
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)
- **RAM**: 16GB
- **Storage**: 10GB+ for extended datasets
- **OS**: Ubuntu 20.04+ or macOS 12+

## 📦 Installation <a name="installation"></a>

### Step 1: Clone the Repository

```bash
git clone https://github.com/lintsinghua/VerifyVision-Pro.git
cd VerifyVision-Pro
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# For macOS/Linux
python -m venv imgvenv
source imgvenv/bin/activate

# For Windows
python -m venv imgvenv
imgvenv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check if PyTorch is properly installed with CUDA (if available)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Optional: GPU Setup

If you have an NVIDIA GPU, ensure you have installed the appropriate CUDA toolkit and cuDNN versions compatible with your PyTorch installation.

## 🚀 Quick Start <a name="quick-start"></a>

Follow this guide to quickly set up and run the VerifyVision-Pro system:

### Step 1: Generate Test Data 🎲

First, generate sample images for testing the system:

```bash
python generate_test_images.py
```

This creates 20 real images and 20 fake images in the respective data directories.

### Step 2: Preprocess Images 🖌️

Prepare the images for model training:

```bash
# Process real images
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224

# Process fake images
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

### Step 3: Train a Model 🧠

Train a basic CNN model using the preprocessed data:

```bash
python main.py train \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake \
  --model cnn \
  --pretrained \
  --epochs 5 \
  --batch-size 4 \
  --save-dir models/saved
```

> **Note**: For initial testing, a small number of epochs (5) is sufficient. Increase for better performance.

### Step 4: Launch Web Application 🌐

Start the web interface to interact with your trained model:

```bash
python main.py web \
  --model-path models/saved/best_model.pth \
  --model-name cnn \
  --port 8080 \
  --debug
```

> **Important**: On macOS, port 5000 may be occupied by AirPlay service. Using port 8080 is recommended.

### Step 5: Access the Application 🖱️

Open your browser and visit [http://localhost:8080](http://localhost:8080) to use the system.

## 📊 Data Preparation <a name="data-preparation"></a>

### Obtaining Datasets 📥

Several methods are available to gather data for training and testing:

#### Method 1: Test Data Generation (Recommended for Beginners)

The built-in script generates synthetic data for testing purposes:

```bash
python generate_test_images.py
```

**What it does:**
- Creates `data/real` and `data/fake` directories
- Generates 20 sample real images with random content
- Creates 20 corresponding fake images with manipulations
- Suitable for initial system testing and validation

#### Method 2: Public Datasets

Access information about public image forgery detection datasets:

```bash
python main.py download-info
```

This displays links to valuable datasets commonly used in image forgery detection research, including:
- CASIA v1.0 and v2.0
- Columbia Image Splicing Detection
- CoMoFoD (Copy-Move Forgery Dataset)
- Coverage
- IEEE IFS-TC Image Forensics Challenge Dataset

#### Method 3: Custom Dataset Creation

Build your own dataset by:

1. **Collecting real images**:
   - Place authentic images in `data/real` directory
   - Use personal photos or public domain images
   - Ensure diversity in content, lighting, and source devices

2. **Creating fake images**:
   ```bash
   python main.py create-fake \
     --real-dir data/real \
     --fake-dir data/fake \
     --method splice \
     --num-images 1000
   ```

**Available forgery methods:**
- `splice`: Combines regions from different images
- `copy`: Duplicates regions within the same image
- `noise`: Adds localized noise to create inconsistencies
- `color`: Manipulates color properties in specific regions

### Preprocessing Data 🔄

Before training, images need to be preprocessed for consistency:

```bash
python main.py preprocess \
  --input-dir data/real \
  --output-dir data/processed/real \
  --target-size 224 224 \
  --max-images 5000
```

**Preprocessing operations include:**
- Resizing to uniform dimensions
- Normalization
- Optional augmentation (rotation, flipping, etc.)
- Format standardization
- Optional color space conversion

**Parameters:**
- `--input-dir`: Source directory containing images
- `--output-dir`: Destination for processed images
- `--target-size`: Output dimensions (width height)
- `--max-images`: Limit number of images to process (optional)
- `--augment`: Apply data augmentation (optional)

## 🏋️‍♂️ Model Training <a name="model-training"></a>

### Training a Model from Scratch

VerifyVision-Pro supports training various deep learning models for image forgery detection:

```bash
python main.py train \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake \
  --model efficientnet_b0 \
  --pretrained \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --save-dir models/saved \
  --early-stopping \
  --patience 5
```

### Available Models

The system implements several state-of-the-art architectures:

| Model | Description | Parameters | Suitable For |
|-------|-------------|------------|--------------|
| `cnn` | Custom CNN | ~500K | Quick testing, limited data |
| `resnet18` | ResNet-18 | ~11M | Small to medium datasets |
| `resnet50` | ResNet-50 | ~25M | Medium datasets |
| `efficientnet_b0` | EfficientNet-B0 | ~5M | Balanced performance |
| `xception` | Xception | ~22M | Advanced features |

### Training Parameters

The training module offers comprehensive customization:

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--real-dir` | Real image directory | - | Required |
| `--fake-dir` | Fake image directory | - | Required |
| `--model` | Model architecture | `efficientnet_b0` | See available models |
| `--pretrained` | Use pretrained weights | `False` | Flag |
| `--epochs` | Training epochs | `30` | |
| `--batch-size` | Batch size | `32` | Reduce for less memory |
| `--learning-rate` | Learning rate | `0.001` | |
| `--weight-decay` | L2 regularization | `0.0001` | |
| `--save-dir` | Save directory | `models/saved` | |
| `--early-stopping` | Enable early stopping | `False` | Flag |
| `--patience` | Epochs for early stopping | `5` | |
| `--validation-split` | Validation data ratio | `0.2` | |

### Training Process

During training, the system:

1. Splits data into training and validation sets
2. Loads or initializes the selected model architecture
3. Applies transfer learning if pretrained weights are requested
4. Optimizes using Adam optimizer with specified learning rate
5. Implements learning rate scheduling for better convergence
6. Monitors validation metrics to prevent overfitting
7. Saves the best-performing model based on validation accuracy
8. Generates training curves and performance statistics

### Advanced Training Features

- **Early Stopping**: Automatically stops training when performance plateaus
- **Learning Rate Scheduling**: Reduces learning rate when progress stalls
- **Checkpointing**: Saves model at regular intervals during training
- **Mixed Precision**: Uses FP16 training when supported by hardware
- **Gradient Clipping**: Prevents exploding gradients
- **Data Augmentation**: Optional real-time augmentation during training

## 📏 Model Evaluation <a name="model-evaluation"></a>

### Evaluating Model Performance

After training, assess your model's performance using:

```bash
python main.py evaluate \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake \
  --model efficientnet_b0 \
  --checkpoint models/saved/best_model.pth \
  --results-dir results \
  --confusion-matrix \
  --roc-curve
```

### Evaluation Metrics

The evaluation module provides comprehensive performance metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| Accuracy | Overall correct predictions | 0-1 |
| Precision | True positives / predicted positives | 0-1 |
| Recall | True positives / actual positives | 0-1 |
| F1 Score | Harmonic mean of precision & recall | 0-1 |
| AUC-ROC | Area Under ROC Curve | 0-1 |
| Confusion Matrix | Visualization of predictions vs. ground truth | - |

### Advanced Evaluation Features

- **Per-class Analysis**: Detailed metrics for real and fake classes
- **Confidence Distribution**: Histogram of prediction confidences
- **Failure Analysis**: Examination of misclassified samples
- **Feature Visualization**: Activation maps showing influential regions
- **Cross-validation**: Optional k-fold cross-validation for robust evaluation

### Interpreting Results

The evaluation results help understand:

- How well the model generalizes to unseen data
- Whether it's biased toward a particular class
- Types of images that cause detection failures
- Confidence level in predictions
- Areas for potential improvement

## 🌐 Web Application <a name="web-application"></a>

### Launching the Web Interface

Start the web application to interact with your trained model:

```bash
python main.py web \
  --model-path models/saved/best_model.pth \
  --model-name efficientnet_b0 \
  --port 8080 \
  --host 0.0.0.0 \
  --debug
```

### Web Application Features

The VerifyVision-Pro web interface provides:

- **User-friendly Upload**: Simple drag-and-drop or file selection interface
- **Real-time Analysis**: Immediate processing and results display
- **Visual Feedback**: Clear indication of authenticity with confidence scores
- **Heatmap Visualization**: Optional visualization of suspicious regions
- **Result History**: Session-based history of analyzed images
- **Responsive Design**: Works on desktop and mobile devices

### Setup Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--model-path` | Path to model file | - | Required |
| `--model-name` | Model architecture | - | Required |
| `--port` | Server port | `5000` | Use `8080` on macOS |
| `--host` | Host address | `127.0.0.1` | Use `0.0.0.0` for external access |
| `--debug` | Enable debug mode | `False` | Flag |
| `--max-size` | Max upload size (MB) | `5` | |
| `--threshold` | Detection threshold | `0.5` | Range: 0-1 |

### Using the Web Application 💻

1. **Upload an Image**:
   - Click "Choose File" or drag-and-drop an image onto the upload area
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 5MB (configurable)

2. **Analyze the Image**:
   - Click "Upload & Detect" button
   - The system processes the image through the model

3. **View Results**:
   - Real/Fake classification is displayed
   - Confidence score indicates detection certainty
   - Optional heatmap visualization highlights suspicious regions
   - Additional metadata shows image properties

4. **Interpret Results**:
   - Higher confidence scores indicate greater certainty
   - Scores near 0.5 indicate uncertainty
   - Consider using multiple models for ambiguous cases

### Deployment Options

For production deployment, consider:

- **Nginx/Apache**: Set up reverse proxy for better security and performance
- **Docker**: Containerized deployment for consistent environment
- **Cloud Platforms**: AWS, Google Cloud, or Azure for scalability
- **SSL Certificate**: Enable HTTPS for secure communication
- **Rate Limiting**: Prevent abuse of the service

## 🔧 Technical Implementation <a name="technical-implementation"></a>

### Core Technologies

VerifyVision-Pro is built on modern technologies for reliable performance:

#### Data Processing
- **OpenCV**: Image loading, preprocessing, and manipulation
- **PIL (Pillow)**: Image format handling and transformations
- **Albumentations**: Advanced data augmentation pipeline
- **NumPy**: Efficient numerical operations on image data

#### Deep Learning Framework
- **PyTorch**: Primary deep learning framework
- **TorchVision**: Pre-trained models and dataset utilities
- **CUDA**: GPU acceleration for training and inference
- **torchinfo**: Model architecture visualization and analysis

#### Model Architectures
- **EfficientNet**: Resource-efficient convolutional architecture
- **ResNet**: Deep residual networks with skip connections
- **Xception**: Depthwise separable convolutions for efficiency
- **Custom CNN**: Lightweight architecture for basic detection

#### Web Framework
- **Flask**: Lightweight web server implementation
- **Werkzeug**: WSGI utility library for web applications
- **Jinja2**: Templating engine for HTML generation
- **Flask-WTF**: Form handling and validation

#### Frontend
- **Bootstrap**: Responsive design framework
- **JavaScript**: Dynamic client-side functionality
- **Chart.js**: Interactive visualization of results
- **Dropzone.js**: Enhanced file upload experience

### Implementation Details

#### Model Architecture Design

The system implements a two-class classification approach with:

- **Feature Extraction**: Convolutional layers capture spatial features
- **Feature Aggregation**: Pooling operations aggregate local information
- **Classification Head**: Fully connected layers for final prediction
- **Transfer Learning**: Adaptation of pre-trained networks
- **Domain-specific Features**: Custom layers for forgery detection

#### Training Pipeline

The training system implements:

- **Dataset Management**: Custom PyTorch datasets for efficient loading
- **Balanced Sampling**: Ensures equal representation of classes
- **Augmentation Strategy**: Applied during training for robustness
- **Mixed Precision**: FP16 for faster training where supported
- **Distributed Training**: Optional multi-GPU support

#### Inference Pipeline

The inference system includes:

- **Preprocessing**: Consistent with training pipeline
- **Batched Processing**: Efficient handling of multiple images
- **Model Ensemble**: Optional combination of multiple models
- **Post-processing**: Confidence calibration and thresholding
- **Visualization**: Generation of heatmaps for interpretability

## 🔬 Advanced Usage <a name="advanced-usage"></a>

### Custom Model Development

Extend VerifyVision-Pro with custom model architectures:

1. **Adding a New Model**:
   
   Modify `src/models/models.py` to include your architecture:

   ```python
   class CustomModel(nn.Module):
       def __init__(self, num_classes=2, pretrained=False):
           super(CustomModel, self).__init__()
           # Define your model architecture here
           
       def forward(self, x):
           # Define forward pass
           return x
   ```

2. **Registering the Model**:
   
   Add your model to the model factory:

   ```python
   def get_model(name, num_classes=2, pretrained=False):
       models = {
           # Existing models
           'custom_model': CustomModel,
       }
       return models[name](num_classes=num_classes, pretrained=pretrained)
   ```

3. **Using Your Model**:
   
   ```bash
   python main.py train \
     --real-dir data/processed/real \
     --fake-dir data/processed/fake \
     --model custom_model \
     --epochs 30
   ```

### Advanced Dataset Techniques

Enhance model performance with advanced dataset handling:

#### Synthetic Data Generation

Create additional training data using generative methods:

```bash
python main.py generate-synthetic \
  --base-images data/real \
  --output-dir data/synthetic \
  --count 1000 \
  --techniques "copy,splice,removal,noise"
```

#### Cross-dataset Validation

Test model generalization across different datasets:

```bash
python main.py cross-validate \
  --train-real data/datasetA/real \
  --train-fake data/datasetA/fake \
  --test-real data/datasetB/real \
  --test-fake data/datasetB/fake \
  --model efficientnet_b0
```

#### Active Learning

Implement active learning to prioritize labeling efforts:

```bash
python main.py active-learning \
  --unlabeled data/unlabeled \
  --labeled data/labeled \
  --model-path models/saved/model.pth \
  --selection-method "entropy" \
  --batch-size 100
```

### Model Interpretation

Understand model decisions with advanced visualization:

```bash
python main.py interpret \
  --image path/to/image.jpg \
  --model-path models/saved/model.pth \
  --method "gradcam" \
  --output-dir visualizations
```

Available interpretation methods:
- `gradcam`: Gradient-weighted Class Activation Mapping
- `lime`: Local Interpretable Model-agnostic Explanations
- `shap`: SHapley Additive exPlanations
- `occlusion`: Occlusion sensitivity analysis

## ⚡ Performance Optimization <a name="performance-optimization"></a>

### Hardware Acceleration

Maximize system performance with hardware optimizations:

#### GPU Acceleration

Enable GPU acceleration for faster training and inference:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Train with GPU (automatic if available)
python main.py train --model efficientnet_b0 --batch-size 64 --real-dir data/processed/real --fake-dir data/processed/fake
```

#### Multi-GPU Training

Distribute training across multiple GPUs for larger models:

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py train \
  --distributed \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake \
  --model efficientnet_b0 \
  --batch-size 128
```

#### CPU Optimization

Optimize CPU performance when GPU is unavailable:

```bash
# Set number of CPU threads
python main.py train --num-workers 8 --pin-memory --real-dir data/processed/real --fake-dir data/processed/fake
```

### Memory Optimization

Manage memory usage for efficient processing:

#### Batch Size Adjustment

Adjust batch size based on available memory:

| Hardware | Recommended Batch Size |
|----------|------------------------|
| CPU | 8-16 |
| GPU 4GB VRAM | 16-32 |
| GPU 8GB VRAM | 32-64 |
| GPU 16GB+ VRAM | 64-128 |

```bash
# Smaller batch size for limited memory
python main.py train --batch-size 8 --real-dir data/processed/real --fake-dir data/processed/fake

# Larger batch size for high-end systems
python main.py train --batch-size 128 --real-dir data/processed/real --fake-dir data/processed/fake
```

#### Gradient Accumulation

Train with large effective batch sizes on limited memory:

```bash
python main.py train \
  --batch-size 16 \
  --gradient-accumulation 4 \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake
```

This simulates a batch size of 64 (16 × 4) while only requiring memory for 16 samples.

### Inference Optimization

Speed up production deployment:

#### Model Quantization

Reduce model size and increase inference speed:

```bash
python main.py quantize \
  --model-path models/saved/best_model.pth \
  --quantized-model-path models/saved/quantized_model.pth \
  --calibration-images data/processed/real
```

This reduces model size by up to 75% and increases inference speed by 2-4x.

#### Batch Inference

Process multiple images simultaneously:

```bash
python main.py batch-inference \
  --input-dir data/test \
  --output-file results.csv \
  --model-path models/saved/best_model.pth \
  --batch-size 32
```

#### Model Pruning

Remove unnecessary connections for faster inference:

```bash
python main.py prune \
  --model-path models/saved/best_model.pth \
  --pruned-model-path models/saved/pruned_model.pth \
  --prune-ratio 0.3
```

## 🔨 Troubleshooting <a name="troubleshooting"></a>

### Common Issues and Solutions

This section addresses frequently encountered problems:

### 🔄 Installation Issues

#### CUDA Compatibility Problems

**Symptoms**: PyTorch installation succeeds but CUDA is not detected or crashes occur during GPU operations.

**Solution**:
1. Ensure compatible versions:
   ```bash
   # Check CUDA version
   nvcc --version
   
   # Install compatible PyTorch version
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Verify installation:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

#### Package Dependency Conflicts

**Symptoms**: `pip install` fails with dependency conflicts.

**Solution**:
1. Create a fresh virtual environment:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   ```

2. Install dependencies one by one:
   ```bash
   pip install numpy
   pip install torch torchvision
   pip install -r requirements.txt
   ```

### 🖥️ Runtime Issues

#### Port Occupation on macOS

**Symptoms**: Web application fails to start with "Address already in use" error.

**Solution**:
1. Use a different port:
   ```bash
   python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080
   ```

2. Or find and kill the process using port 5000:
   ```bash
   sudo lsof -i :5000
   kill -9 <PID>
   ```

#### Out of Memory (OOM) Errors

**Symptoms**: Training crashes with "CUDA out of memory" or system memory errors.

**Solution**:
1. Reduce batch size:
   ```bash
   python main.py train --batch-size 4 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. Use gradient accumulation:
   ```bash
   python main.py train --batch-size 2 --gradient-accumulation 8 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. Use a smaller model:
   ```bash
   python main.py train --model resnet18 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

#### Empty Dataset Errors

**Symptoms**: Training fails with "dataset is empty" errors.

**Solution**:
1. Verify directory paths:
   ```bash
   ls -la data/processed/real data/processed/fake
   ```

2. Check file formats (should be .jpg, .jpeg, or .png):
   ```bash
   find data/processed/real -type f | grep -v -E '\.(jpg|jpeg|png)$'
   ```

3. Generate test data to verify system:
   ```bash
   python generate_test_images.py
   ```

### 🏋️‍♂️ Training Issues

#### Poor Model Performance

**Symptoms**: Model achieves low accuracy or doesn't improve during training.

**Solution**:
1. Increase training duration:
   ```bash
   python main.py train --epochs 50 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. Try different models:
   ```bash
   python main.py train --model efficientnet_b0 --pretrained --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. Ensure balanced dataset:
   ```bash
   python main.py analyze-dataset --real-dir data/processed/real --fake-dir data/processed/fake
   ```

4. Enable data augmentation:
   ```bash
   python main.py train --augmentation --real-dir data/processed/real --fake-dir data/processed/fake
   ```

#### Training Plateaus

**Symptoms**: Validation accuracy stops improving early in training.

**Solution**:
1. Adjust learning rate:
   ```bash
   python main.py train --learning-rate 0.0001 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. Implement learning rate scheduling:
   ```bash
   python main.py train --scheduler cosine --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. Try different optimizers:
   ```bash
   python main.py train --optimizer adamw --real-dir data/processed/real --fake-dir data/processed/fake
   ```

#### Overfitting

**Symptoms**: Training accuracy is high but validation accuracy is low.

**Solution**:
1. Add regularization:
   ```bash
   python main.py train --weight-decay 0.001 --dropout 0.3 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. Use early stopping:
   ```bash
   python main.py train --early-stopping --patience 5 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. Increase dataset size or diversity.

## 📝 Notes & Best Practices <a name="notes"></a>

### Practical Recommendations

#### Dataset Quality

The quality of training data directly impacts model performance:

- **Size**: 1,000+ images per class minimum for good performance
- **Balance**: Maintain equal numbers of real and fake images
- **Diversity**: Include various image sources, lighting conditions, and content
- **Authenticity**: Ensure "real" images are truly unmanipulated
- **Realism**: Create forgeries that represent realistic manipulation methods
- **Metadata**: Preserve relevant metadata (camera model, editing software, etc.)

#### Model Selection

Choose models based on your specific requirements:

| Priority | Recommended Model |
|----------|-------------------|
| Speed | `cnn` or `resnet18` |
| Accuracy | `efficientnet_b0` or `xception` |
| Balance | `resnet18` or `efficientnet_b0` |
| Limited Data | `cnn` with heavy augmentation |
| Production | Ensemble of multiple models |

#### Deployment Considerations

For real-world deployment:

- **Security**: Implement rate limiting and file validation
- **Scalability**: Use load balancing for high-traffic applications
- **Privacy**: Consider local processing for sensitive materials
- **Transparency**: Communicate confidence levels and limitations
- **Updates**: Regularly retrain with new forgery techniques
- **Fallback**: Have human review for critical or ambiguous cases

#### Detection Limitations

Be aware of system limitations:

- Detection accuracy varies by forgery type and quality
- Advanced AI-generated images may require specialized models
- Very small manipulations might be missed
- Results should be treated as probabilistic, not definitive
- System should be part of a broader verification strategy

## 🤝 Contributing <a name="contributing"></a>

We welcome contributions to VerifyVision-Pro! Here's how you can help:

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Include detailed steps to reproduce the issue
- Attach sample images when relevant (ensure you have rights to share)
- Specify your environment (OS, Python version, etc.)

### Development Process

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests**:
   ```bash
   python -m pytest tests/
   ```
5. **Submit a pull request**

### Contribution Areas

We particularly welcome contributions in:

- **New Models**: Implementations of state-of-the-art architectures
- **Detection Methods**: Novel approaches to identifying manipulations
- **UI Improvements**: Enhancing the web interface and visualization
- **Performance Optimization**: Improving speed and resource usage
- **Documentation**: Tutorials, examples, and clarifications
- **Localization**: Translations of documentation and interface

### Code Style

Please follow these guidelines:

- PEP 8 compliant Python code
- Docstrings for all functions, classes, and modules
- Type hints where appropriate
- Comprehensive comments for complex logic
- Unit tests for new functionality

## 📄 License <a name="license"></a>

VerifyVision-Pro is released under the MIT License.

### MIT License

```
Copyright (c) 2023 VerifyVision-Pro Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Components

This project incorporates components from third-party open source projects:

- PyTorch (BSD License)
- Flask (BSD License)
- TorchVision (BSD License)
- OpenCV (Apache 2.0 License)
- Bootstrap (MIT License)
- Various other packages as listed in requirements.txt

---

<a name="chinese-documentation"></a>
# 中文文档 🌏

## 📋 目录

- [概述](#概述)
- [项目结构](#项目结构)
- [系统要求](#系统要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [Web应用](#web应用)
- [技术实现](#技术实现)
- [高级用法](#高级用法)
- [性能优化](#性能优化)
- [常见问题解决](#常见问题解决)
- [注意事项](#注意事项)
- [参与贡献](#参与贡献)
- [许可证](#许可证)
