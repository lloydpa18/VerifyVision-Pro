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
Copyright (c) 2025 VerifyVision-Pro Contributors

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

## 🔭 概述 <a name="概述"></a>

VerifyVision-Pro是一个综合性的基于深度学习的图像伪造检测系统，能够高精度地识别各种图像篡改。系统集成了强大的数据处理流程、先进的深度学习模型和直观的Web界面，实现实时检测功能。

### 🌟 主要特点

- **多模型支持**：实现多种架构（EfficientNet、ResNet、Xception、CNN）
- **完整流程**：从数据准备到部署的全流程解决方案
- **用户友好界面**：基于Web的界面，便于与系统交互
- **详细分析**：提供置信度评分和结果可视化
- **优化性能**：支持CPU和GPU推理加速

## 📁 项目结构 <a name="项目结构"></a>

```
VerifyVision-Pro/
│
├── data/                      # 数据目录
│   ├── real/                  # 真实图像
│   ├── fake/                  # 伪造图像
│   └── processed/             # 预处理后的图像
│
├── models/                    # 模型目录（已被git忽略）
│   └── saved/                 # 保存的模型权重
│
├── src/                       # 源代码
│   ├── data_utils/            # 数据处理工具
│   │   ├── dataset.py         # 数据集类
│   │   └── data_processor.py  # 数据预处理工具
│   │
│   ├── models/                # 模型定义
│   │   └── models.py          # 深度学习模型实现
│   │
│   ├── training/              # 训练相关
│   │   ├── train.py           # 训练脚本
│   │   └── evaluate.py        # 评估脚本
│   │
│   └── web/                   # Web应用
│       └── app.py             # Flask应用
│
├── static/                    # 静态资源
│   ├── css/                   # CSS样式
│   │   └── style.css          # 自定义样式
│   │
│   ├── js/                    # JavaScript
│   │   └── main.js            # 主JS文件
│   │
│   └── uploads/               # 用户上传的图像
│
├── templates/                 # HTML模板
│   ├── base.html              # 基础模板
│   ├── index.html             # 首页
│   ├── result.html            # 结果页面
│   └── about.html             # 关于页面
│
├── generate_test_images.py    # 测试图像生成脚本
├── main.py                    # 项目主入口程序
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 💻 系统要求 <a name="系统要求"></a>

### 最低配置

- **Python**: 3.7+
- **PyTorch**: 2.0+
- **内存**: 4GB（仅CPU），8GB（带GPU）
- **存储**: 代码和基本数据集需要1GB
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### 推荐配置

- **Python**: 3.9+
- **PyTorch**: 2.0+（带CUDA支持）
- **GPU**: NVIDIA GPU，支持CUDA（8GB+显存）
- **内存**: 16GB
- **存储**: 扩展数据集需要10GB+
- **操作系统**: Ubuntu 20.04+或macOS 12+

## 📦 安装 <a name="安装"></a>

### 步骤1：克隆仓库

```bash
git clone https://github.com/lintsinghua/VerifyVision-Pro.git
cd VerifyVision-Pro
```

### 步骤2：创建虚拟环境（推荐）

```bash
# macOS/Linux
python -m venv imgvenv
source imgvenv/bin/activate

# Windows
python -m venv imgvenv
imgvenv\Scripts\activate
```

### 步骤3：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤4：验证安装

```bash
# 检查PyTorch是否正确安装并支持CUDA（如果可用）
python -c "import torch; print('CUDA可用：', torch.cuda.is_available())"
```

### 可选：GPU配置

如果您有NVIDIA GPU，请确保安装了与您的PyTorch版本兼容的CUDA工具包和cuDNN。

## 🚀 快速开始 <a name="快速开始"></a>

按照以下指南快速设置和运行VerifyVision-Pro系统：

### 步骤1：生成测试数据 🎲

首先，为系统测试生成样本图像：

```bash
python generate_test_images.py
```

这将在相应的数据目录中创建20张真实图像和20张伪造图像。

### 步骤2：预处理图像 🖌️

准备用于模型训练的图像：

```bash
# 处理真实图像
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224

# 处理伪造图像
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

### 步骤3：训练模型 🧠

使用预处理后的数据训练基本CNN模型：

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

> **注意**：初始测试时，少量的epochs（如5）就足够了。若要提高性能，可增加训练周期。

### 步骤4：启动Web应用 🌐

启动Web界面以与训练好的模型交互：

```bash
python main.py web \
  --model-path models/saved/best_model.pth \
  --model-name cnn \
  --port 8080 \
  --debug
```

> **重要**：在macOS上，端口5000可能被AirPlay服务占用，建议使用端口8080。

### 步骤5：访问应用 🖱️

打开浏览器访问[http://localhost:8080](http://localhost:8080)即可使用系统。

## 📊 数据准备 <a name="数据准备"></a>

### 获取数据集 📥

有多种方法可以收集训练和测试数据：

#### 方法1：测试数据生成（推荐初学者使用）

内置脚本可生成用于测试的合成数据：

```bash
python generate_test_images.py
```

**功能说明：**
- 创建`data/real`和`data/fake`目录
- 生成20张具有随机内容的真实图像
- 创建20张对应的伪造图像
- 适用于初始系统测试和验证

#### 方法2：公开数据集

获取关于公开图像伪造检测数据集的信息：

```bash
python main.py download-info
```

这会显示图像伪造检测研究中常用的数据集链接，包括：
- CASIA v1.0和v2.0
- Columbia图像拼接检测
- CoMoFoD（拷贝-移动伪造数据集）
- Coverage
- IEEE IFS-TC图像取证挑战数据集

#### 方法3：自定义数据集创建

通过以下方法构建自己的数据集：

1. **收集真实图像**：
   - 将真实图像放入`data/real`目录
   - 使用个人照片或公共领域图像
   - 确保内容、光线条件和来源设备多样化

2. **创建伪造图像**：
   ```bash
   python main.py create-fake \
     --real-dir data/real \
     --fake-dir data/fake \
     --method splice \
     --num-images 1000
   ```

**可用伪造方法：**
- `splice`：组合来自不同图像的区域
- `copy`：复制同一图像内的区域
- `noise`：添加局部噪声以创建不一致
- `color`：在特定区域操作颜色属性

### 预处理数据 🔄

训练前，需要对图像进行预处理以保持一致性：

```bash
python main.py preprocess \
  --input-dir data/real \
  --output-dir data/processed/real \
  --target-size 224 224 \
  --max-images 5000
```

**预处理操作包括：**
- 调整为统一尺寸
- 标准化
- 可选的数据增强（旋转、翻转等）
- 格式标准化
- 可选的色彩空间转换

**参数说明：**
- `--input-dir`：源图像目录
- `--output-dir`：处理后图像的目标目录
- `--target-size`：输出尺寸（宽度 高度）
- `--max-images`：限制处理的图像数量（可选）
- `--augment`：应用数据增强（可选）

## 🏋️‍♂️ 模型训练 <a name="模型训练"></a>

### 从头开始训练模型

VerifyVision-Pro支持训练各种深度学习模型进行图像伪造检测：

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

### 可用模型

系统实现了多种最先进的架构：

| 模型 | 描述 | 参数数量 | 适用场景 |
|-------|-------------|------------|--------------|
| `cnn` | 自定义CNN | ~500K | 快速测试，有限数据 |
| `resnet18` | ResNet-18 | ~11M | 小型到中型数据集 |
| `resnet50` | ResNet-50 | ~25M | 中型数据集 |
| `efficientnet_b0` | EfficientNet-B0 | ~5M | 平衡性能 |
| `xception` | Xception | ~22M | 高级特征 |

### 训练参数

训练模块提供全面的自定义选项：

| 参数 | 描述 | 默认值 | 备注 |
|-----------|-------------|---------|-------|
| `--real-dir` | 真实图像目录 | - | 必需 |
| `--fake-dir` | 伪造图像目录 | - | 必需 |
| `--model` | 模型架构 | `efficientnet_b0` | 查看可用模型 |
| `--pretrained` | 使用预训练权重 | `False` | 标志 |
| `--epochs` | 训练周期 | `30` | |
| `--batch-size` | 批次大小 | `32` | 减小以降低内存占用 |
| `--learning-rate` | 学习率 | `0.001` | |
| `--weight-decay` | L2正则化 | `0.0001` | |
| `--save-dir` | 保存目录 | `models/saved` | |
| `--early-stopping` | 启用早停 | `False` | 标志 |
| `--patience` | 早停周期数 | `5` | |
| `--validation-split` | 验证数据比例 | `0.2` | |

### 训练过程

训练过程中，系统会：

1. 将数据分割为训练集和验证集
2. 加载或初始化所选模型架构
3. 如果请求了预训练权重，应用迁移学习
4. 使用指定学习率的Adam优化器进行优化
5. 实现学习率调度以获得更好的收敛性
6. 监控验证指标以防止过拟合
7. 根据验证精度保存表现最佳的模型
8. 生成训练曲线和性能统计

### 高级训练功能

- **早停**：当性能达到平台期时自动停止训练
- **学习率调度**：当进度停滞时降低学习率
- **检查点**：在训练期间定期保存模型
- **混合精度**：在硬件支持时使用FP16训练
- **梯度裁剪**：防止梯度爆炸
- **数据增强**：训练期间可选的实时增强

## 📏 模型评估 <a name="模型评估"></a>

### 评估模型性能

训练后，使用以下命令评估模型性能：

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

### 评估指标

评估模块提供全面的性能指标：

| 指标 | 描述 | 范围 |
|--------|-------------|-------|
| 准确率 | 总体正确预测比例 | 0-1 |
| 精确率 | 真阳性/预测阳性 | 0-1 |
| 召回率 | 真阳性/实际阳性 | 0-1 |
| F1分数 | 精确率和召回率的调和平均 | 0-1 |
| AUC-ROC | ROC曲线下面积 | 0-1 |
| 混淆矩阵 | 预测与真实值的可视化 | - |

### 高级评估功能

- **分类别分析**：真实和伪造类别的详细指标
- **置信度分布**：预测置信度的直方图
- **失败分析**：对错误分类样本的检查
- **特征可视化**：显示影响区域的激活图
- **交叉验证**：可选的k折交叉验证以获得稳健评估

### 解读结果

评估结果有助于理解：

- 模型对未见数据的泛化能力
- 模型是否偏向特定类别
- 导致检测失败的图像类型
- 预测的置信度水平
- 可能的改进领域

## 🌐 Web应用 <a name="web应用"></a>

### 启动Web界面

启动Web应用以与训练好的模型交互：

```bash
python main.py web \
  --model-path models/saved/best_model.pth \
  --model-name efficientnet_b0 \
  --port 8080 \
  --host 0.0.0.0 \
  --debug
```

### Web应用功能

VerifyVision-Pro的Web界面提供：

- **用户友好的上传**：简单的拖放或文件选择界面
- **实时分析**：即时处理和结果显示
- **视觉反馈**：清晰显示真伪结果和置信度分数
- **热力图可视化**：可选的可疑区域可视化
- **结果历史**：基于会话的分析图像历史
- **响应式设计**：适用于桌面和移动设备

### 设置参数

| 参数 | 描述 | 默认值 | 备注 |
|-----------|-------------|---------|-------|
| `--model-path` | 模型文件路径 | - | 必需 |
| `--model-name` | 模型架构 | - | 必需 |
| `--port` | 服务器端口 | `5000` | macOS上使用`8080` |
| `--host` | 主机地址 | `127.0.0.1` | 外部访问使用`0.0.0.0` |
| `--debug` | 启用调试模式 | `False` | 标志 |
| `--max-size` | 最大上传大小(MB) | `5` | |
| `--threshold` | 检测阈值 | `0.5` | 范围：0-1 |

### 使用Web应用 💻

1. **上传图像**：
   - 点击"选择文件"或将图像拖放到上传区域
   - 支持的格式：JPG、JPEG、PNG
   - 最大文件大小：5MB（可配置）

2. **分析图像**：
   - 点击"上传并检测"按钮
   - 系统通过模型处理图像

3. **查看结果**：
   - 显示真实/伪造分类结果
   - 置信度分数表示检测确定性
   - 可选的热力图可视化突出显示可疑区域
   - 附加元数据显示图像属性

4. **解读结果**：
   - 更高的置信度分数表示更大的确定性
   - 接近0.5的分数表示不确定性
   - 对于模糊的情况，考虑使用多个模型

### 部署选项

对于生产部署，考虑以下方案：

- **Nginx/Apache**：设置反向代理以提高安全性和性能
- **Docker**：容器化部署以保持环境一致性
- **云平台**：AWS、Google Cloud或Azure以实现可扩展性
- **SSL证书**：启用HTTPS以进行安全通信
- **访问限制**：防止服务滥用

## 🔧 技术实现 <a name="技术实现"></a>

### 核心技术

VerifyVision-Pro基于现代技术构建，以实现可靠的性能：

#### 数据处理
- **OpenCV**：图像加载、预处理和操作
- **PIL (Pillow)**：图像格式处理和转换
- **Albumentations**：高级数据增强流程
- **NumPy**：图像数据的高效数值运算

#### 深度学习框架
- **PyTorch**：主要深度学习框架
- **TorchVision**：预训练模型和数据集实用工具
- **CUDA**：用于训练和推理的GPU加速
- **torchinfo**：模型架构可视化和分析

#### 模型架构
- **EfficientNet**：资源高效的卷积架构
- **ResNet**：带跳跃连接的深度残差网络
- **Xception**：深度可分离卷积以提高效率
- **自定义CNN**：用于基本检测的轻量级架构

#### Web框架
- **Flask**：轻量级Web服务器实现
- **Werkzeug**：Web应用的WSGI实用工具库
- **Jinja2**：HTML生成的模板引擎
- **Flask-WTF**：表单处理和验证

#### 前端
- **Bootstrap**：响应式设计框架
- **JavaScript**：动态客户端功能
- **Chart.js**：结果的交互式可视化
- **Dropzone.js**：增强的文件上传体验

### 实现细节

#### 模型架构设计

系统实现了两类分类方法，包含：

- **特征提取**：卷积层捕获空间特征
- **特征聚合**：池化操作聚合局部信息
- **分类头**：全连接层用于最终预测
- **迁移学习**：预训练网络的适应
- **领域特定特征**：用于伪造检测的自定义层

#### 训练流程

训练系统实现：

- **数据集管理**：自定义PyTorch数据集用于高效加载
- **平衡采样**：确保类别平等表示
- **增强策略**：在训练期间应用以提高鲁棒性
- **混合精度**：在支持的情况下使用FP16加速训练
- **分布式训练**：可选的多GPU支持

#### 推理流程

推理系统包括：

- **预处理**：与训练流程一致
- **批处理**：高效处理多个图像
- **模型集成**：可选的多模型组合
- **后处理**：置信度校准和阈值处理
- **可视化**：生成解释性热力图

## 🔬 高级用法 <a name="高级用法"></a>

### 自定义模型开发

通过自定义模型架构扩展VerifyVision-Pro：

1. **添加新模型**：
   
   修改`src/models/models.py`以包括您的架构：

   ```python
   class CustomModel(nn.Module):
       def __init__(self, num_classes=2, pretrained=False):
           super(CustomModel, self).__init__()
           # 定义您的模型架构
           
       def forward(self, x):
           # 定义前向传播
           return x
   ```

2. **注册模型**：
   
   将您的模型添加到模型工厂：

   ```python
   def get_model(name, num_classes=2, pretrained=False):
       models = {
           # 现有模型
           'custom_model': CustomModel,
       }
       return models[name](num_classes=num_classes, pretrained=pretrained)
   ```

3. **使用您的模型**：
   
   ```bash
   python main.py train \
     --real-dir data/processed/real \
     --fake-dir data/processed/fake \
     --model custom_model \
     --epochs 30
   ```

### 高级数据集技术

通过高级数据集处理增强模型性能：

#### 合成数据生成

使用生成方法创建额外的训练数据：

```bash
python main.py generate-synthetic \
  --base-images data/real \
  --output-dir data/synthetic \
  --count 1000 \
  --techniques "copy,splice,removal,noise"
```

#### 跨数据集验证

测试模型在不同数据集间的泛化能力：

```bash
python main.py cross-validate \
  --train-real data/datasetA/real \
  --train-fake data/datasetA/fake \
  --test-real data/datasetB/real \
  --test-fake data/datasetB/fake \
  --model efficientnet_b0
```

#### 主动学习

实现主动学习以优先标注努力：

```bash
python main.py active-learning \
  --unlabeled data/unlabeled \
  --labeled data/labeled \
  --model-path models/saved/model.pth \
  --selection-method "entropy" \
  --batch-size 100
```

### 模型解释

通过高级可视化理解模型决策：

```bash
python main.py interpret \
  --image path/to/image.jpg \
  --model-path models/saved/model.pth \
  --method "gradcam" \
  --output-dir visualizations
```

可用的解释方法：
- `gradcam`：梯度加权类激活映射
- `lime`：局部可解释的模型不可知解释
- `shap`：Shapley加性解释
- `occlusion`：遮挡敏感性分析

## ⚡ 性能优化 <a name="性能优化"></a>

### 硬件加速

通过硬件优化最大化系统性能：

#### GPU加速

启用GPU加速以实现更快的训练和推理：

```bash
# 检查GPU可用性
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')"

# 使用GPU训练（如果可用则自动使用）
python main.py train --model efficientnet_b0 --batch-size 64 --real-dir data/processed/real --fake-dir data/processed/fake
```

#### 多GPU训练

将训练分布到多个GPU上以处理更大的模型：

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py train \
  --distributed \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake \
  --model efficientnet_b0 \
  --batch-size 128
```

#### CPU优化

在GPU不可用时优化CPU性能：

```bash
# 设置CPU线程数
python main.py train --num-workers 8 --pin-memory --real-dir data/processed/real --fake-dir data/processed/fake
```

### 内存优化

管理内存使用以实现高效处理：

#### 批大小调整

根据可用内存调整批大小：

| 硬件 | 推荐批大小 |
|----------|------------------------|
| CPU | 8-16 |
| GPU 4GB显存 | 16-32 |
| GPU 8GB显存 | 32-64 |
| GPU 16GB+显存 | 64-128 |

```bash
# 内存有限时使用较小批大小
python main.py train --batch-size 8 --real-dir data/processed/real --fake-dir data/processed/fake

# 高端系统使用较大批大小
python main.py train --batch-size 128 --real-dir data/processed/real --fake-dir data/processed/fake
```

#### 梯度累积

在有限内存上使用大的有效批大小进行训练：

```bash
python main.py train \
  --batch-size 16 \
  --gradient-accumulation 4 \
  --real-dir data/processed/real \
  --fake-dir data/processed/fake
```

这模拟了64（16 × 4）的批大小，但只需要16个样本的内存。

### 推理优化

加速生产部署：

#### 模型量化

减少模型大小并提高推理速度：

```bash
python main.py quantize \
  --model-path models/saved/best_model.pth \
  --quantized-model-path models/saved/quantized_model.pth \
  --calibration-images data/processed/real
```

这可将模型大小减少高达75%，并将推理速度提高2-4倍。

#### 批量推理

同时处理多个图像：

```bash
python main.py batch-inference \
  --input-dir data/test \
  --output-file results.csv \
  --model-path models/saved/best_model.pth \
  --batch-size 32
```

#### 模型剪枝

移除不必要的连接以加快推理：

```bash
python main.py prune \
  --model-path models/saved/best_model.pth \
  --pruned-model-path models/saved/pruned_model.pth \
  --prune-ratio 0.3
```

## 🔨 常见问题解决 <a name="常见问题解决"></a>

### 常见问题及解决方案

本节解决常见问题：

#### 🔄 安装问题

##### CUDA兼容性问题

**症状**：PyTorch安装成功但CUDA未被检测到，或在GPU操作期间崩溃。

**解决方案**：
1. 确保版本兼容：
   ```bash
   # 检查CUDA版本
   nvcc --version
   
   # 安装兼容的PyTorch版本
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. 验证安装：
   ```bash
   python -c "import torch; print('CUDA可用：', torch.cuda.is_available())"
   ```

##### 包依赖冲突

**症状**：`pip install`因依赖冲突而失败。

**解决方案**：
1. 创建新的虚拟环境：
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   ```

2. 逐个安装依赖：
   ```bash
   pip install numpy
   pip install torch torchvision
   pip install -r requirements.txt
   ```

#### 🖥️ 运行时问题

##### macOS上的端口占用

**症状**：Web应用启动失败，提示"地址已被使用"错误。

**解决方案**：
1. 使用不同端口：
   ```bash
   python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080
   ```

2. 或找到并终止使用端口5000的进程：
   ```bash
   sudo lsof -i :5000
   kill -9 <PID>
   ```

##### 内存溢出(OOM)错误

**症状**：训练崩溃，出现"CUDA内存不足"或系统内存错误。

**解决方案**：
1. 减小批大小：
   ```bash
   python main.py train --batch-size 4 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. 使用梯度累积：
   ```bash
   python main.py train --batch-size 2 --gradient-accumulation 8 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. 使用更小的模型：
   ```bash
   python main.py train --model resnet18 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

##### 数据集为空错误

**症状**：训练失败，提示"数据集为空"错误。

**解决方案**：
1. 验证目录路径：
   ```bash
   ls -la data/processed/real data/processed/fake
   ```

2. 检查文件格式（应为.jpg、.jpeg或.png）：
   ```bash
   find data/processed/real -type f | grep -v -E '\.(jpg|jpeg|png)$'
   ```

3. 生成测试数据以验证系统：
   ```bash
   python generate_test_images.py
   ```

#### 🏋️‍♂️ 训练问题

##### 模型性能不佳

**症状**：模型准确率低或训练期间没有改善。

**解决方案**：
1. 增加训练时长：
   ```bash
   python main.py train --epochs 50 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. 尝试不同模型：
   ```bash
   python main.py train --model efficientnet_b0 --pretrained --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. 确保数据集平衡：
   ```bash
   python main.py analyze-dataset --real-dir data/processed/real --fake-dir data/processed/fake
   ```

4. 启用数据增强：
   ```bash
   python main.py train --augmentation --real-dir data/processed/real --fake-dir data/processed/fake
   ```

##### 训练平台期

**症状**：验证准确率在训练早期停止改善。

**解决方案**：
1. 调整学习率：
   ```bash
   python main.py train --learning-rate 0.0001 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. 实现学习率调度：
   ```bash
   python main.py train --scheduler cosine --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. 尝试不同优化器：
   ```bash
   python main.py train --optimizer adamw --real-dir data/processed/real --fake-dir data/processed/fake
   ```

##### 过拟合

**症状**：训练准确率高但验证准确率低。

**解决方案**：
1. 添加正则化：
   ```bash
   python main.py train --weight-decay 0.001 --dropout 0.3 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

2. 使用早停：
   ```bash
   python main.py train --early-stopping --patience 5 --real-dir data/processed/real --fake-dir data/processed/fake
   ```

3. 增加数据集大小或多样性。

## 📝 注意事项 <a name="注意事项"></a>

### 实用建议

#### 数据集质量

训练数据的质量直接影响模型性能：

- **规模**：良好性能至少需要每类1,000+图像
- **平衡**：保持真实和伪造图像数量相等
- **多样性**：包括各种图像来源、光照条件和内容
- **真实性**：确保"真实"图像确实未经处理
- **真实感**：创建代表现实操作方法的伪造图像
- **元数据**：保留相关元数据（相机型号、编辑软件等）

#### 模型选择

根据您的具体需求选择模型：

| 优先考虑 | 推荐模型 |
|----------|-------------------|
| 速度 | `cnn`或`resnet18` |
| 准确率 | `efficientnet_b0`或`xception` |
| 平衡性能 | `resnet18`或`efficientnet_b0` |
| 有限数据 | `cnn`配合大量增强 |
| 生产环境 | 多个模型的集成 |

#### 部署考虑因素

对于实际部署：

- **安全性**：实施速率限制和文件验证
- **可扩展性**：对高流量应用使用负载均衡
- **隐私**：考虑敏感材料的本地处理
- **透明度**：传达置信水平和局限性
- **更新**：定期用新的伪造技术重新训练
- **备选方案**：对关键或模糊情况有人工审核

#### 检测局限性

了解系统局限性：

- 检测准确率因伪造类型和质量而异
- 高级AI生成图像可能需要专门模型
- 非常小的操作可能被忽略
- 结果应被视为概率性的，而非确定性的
- 系统应作为更广泛验证策略的一部分

## 🤝 参与贡献 <a name="参与贡献"></a>

我们欢迎对VerifyVision-Pro的贡献！以下是您可以帮助的方式：

### 报告问题

- 使用GitHub issue跟踪器报告bug
- 包括详细的步骤以重现问题
- 必要时附加样本图像（确保您有权共享）
- 指定您的环境（操作系统、Python版本等）

### 开发流程

1. **Fork仓库**
2. **创建特性分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **做出更改**
4. **运行测试**：
   ```bash
   python -m pytest tests/
   ```
5. **提交拉取请求**

### 贡献领域

我们特别欢迎以下领域的贡献：

- **新模型**：最先进架构的实现
- **检测方法**：识别操作的新方法
- **UI改进**：增强Web界面和可视化
- **性能优化**：提高速度和资源使用
- **文档**：教程、示例和说明
- **本地化**：文档和界面的翻译

### 代码风格

请遵循以下准则：

- 符合PEP 8的Python代码
- 所有函数、类和模块的文档字符串
- 适当的类型提示
- 复杂逻辑的全面注释
- 新功能的单元测试

## 📄 许可证 <a name="许可证"></a>

VerifyVision-Pro在MIT许可证下发布。

### MIT许可证

```
版权所有 (c) 2025 VerifyVision-Pro贡献者

特此授予免费许可，任何获得本软件和相关文档文件（"软件"）副本的人，
不受限制地处理本软件，包括但不限于使用、复制、修改、合并、发布、
分发、再许可和/或出售软件副本的权利，并允许向其提供软件的人这样做，
但须符合以下条件：

上述版权声明和本许可声明应包含在软件的所有副本或主要部分中。

本软件按"原样"提供，不提供任何形式的明示或暗示担保，包括但不限于
对适销性、特定用途适用性和非侵权性的担保。在任何情况下，作者或版权
持有人均不对因软件或软件的使用或其他交易而产生的任何索赔、损害或其他
责任负责，无论是契约行为、侵权行为或其他行为。
```

### 第三方组件

本项目包含来自第三方开源项目的组件：

- PyTorch (BSD许可证)
- Flask (BSD许可证)
- TorchVision (BSD许可证)
- OpenCV (Apache 2.0许可证)
- Bootstrap (MIT许可证)
- 其他各种包，如requirements.txt中所列
