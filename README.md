# VerifyVision-Pro: Deep Learning Image Forgery Detection System 🔍🖼️

[English](#english-documentation) | [中文](#chinese-documentation)

<a name="english-documentation"></a>
## English Documentation 🌍

### Overview ℹ️
VerifyVision-Pro is a comprehensive deep learning-based system for detecting image forgeries. The system integrates data processing, model training, and a web interface for real-time detection.

### Project Structure 📂
```
VerifyVision-Pro/
│
├── data/                      # Data directory
│   ├── real/                  # Real images
│   ├── fake/                  # Forged images
│   └── processed/             # Preprocessed images
│
├── models/                    # Model directory
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

### System Requirements 🖥️

- Python 3.7+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation 📦

1. Clone the repository
```bash
git clone https://github.com/lintsinghua/VerifyVision-Pro.git
cd VerifyVision-Pro
```

2. Create a virtual environment (optional)
```bash
python -m venv imgvenv
source imgvenv/bin/activate  # Linux/Mac
imgvenv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Quick Start 🚀

The project provides quick-start scripts for experiencing full functionality:

1. **Generate test data** 🎲
```bash
python generate_test_images.py
```
This will generate 20 real images and 20 fake images in the data directory for subsequent training and testing.

2. **Preprocess images** 🖌️
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

3. **Train the model** 🧠
```bash
python main.py train --real-dir data/processed/real --fake-dir data/processed/fake --model cnn --pretrained --epochs 5 --batch-size 4 --save-dir models/saved
```
Note: A smaller epochs value (e.g., 5) can be used to speed up the training process.

4. **Launch the web application** 🌐
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080 --debug
```
Note: On macOS, port 5000 may be occupied by AirPlay service, so using port 8080 is recommended.

5. **Access the web application** 🖱️
Open your browser and visit http://localhost:8080 to use the system.

### Data Preparation 📊

#### Obtaining Datasets 📥

Data can be obtained through the following methods:

1. **Using the test data generation script** (recommended for first-time users):
```bash
python generate_test_images.py
```
This will automatically create real and fake images for testing system functionality.

2. **Using public dataset information**:
```bash
python main.py download-info
```
The program will display links to available public image forgery detection datasets, which can be downloaded manually.

3. **Creating your own dataset**:
- Collect real images in the `data/real` directory
- Create fake images using the following command:
```bash
python main.py create-fake --real-dir data/real --fake-dir data/fake --method splice --num-images 1000
```

#### Preprocessing Data 🔄

Use the main program interface to process images:
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

Parameters:
- `--input-dir`: Input image directory
- `--output-dir`: Output image directory
- `--target-size`: Target image size, default is 224x224
- `--max-images`: Maximum number of images to process (optional)

### Model Training 🏋️‍♂️

Use the main program interface to train the model:
```bash
python main.py train --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --pretrained --epochs 30 --batch-size 32 --save-dir models/saved
```

Parameter description:
- `--real-dir`: Real image directory
- `--fake-dir`: Fake image directory
- `--model`: Model to use, options include `efficientnet_b0`, `resnet18`, `resnet50`, `xception`, `cnn`
- `--pretrained`: Whether to use pre-trained weights
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--save-dir`: Model save directory

For more parameters, refer to the help information: `python main.py train -h`

### Model Evaluation 📏

Use the main program interface to evaluate model performance:
```bash
python main.py evaluate --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --checkpoint models/saved/best_model.pth --results-dir results
```

Parameter description:
- `--real-dir`: Real image directory
- `--fake-dir`: Fake image directory
- `--model`: Model to use
- `--checkpoint`: Model checkpoint path
- `--results-dir`: Results save directory

### Web Application 🌐

Use the main program interface to start the web application:
```bash
python main.py web --model-path models/saved/best_model.pth --model-name efficientnet_b0 --port 8080
```

Parameter description:
- `--model-path`: Model path
- `--model-name`: Model name
- `--port`: Port number, 8080 is recommended (port 5000 might be occupied on macOS)
- `--debug`: Add this parameter to enable debug mode

#### Using the Web Application 💻

1. Open your browser and visit http://localhost:8080
2. Click the "Choose File" button to upload an image for detection
3. Click the "Upload & Detect" button
4. The system will display the detection result, including the real/fake judgment and the corresponding confidence level

### Technical Implementation 🔧

This project uses the following main technologies:

1. **Data Processing**: OpenCV, PIL, Albumentations
2. **Deep Learning Framework**: PyTorch, TorchVision
3. **Model Architectures**: EfficientNet, ResNet, Xception
4. **Web Framework**: Flask
5. **Frontend**: Bootstrap, JavaScript

### Advanced Usage 🔬

#### Custom Model Training
You can add new model architectures by modifying `src/models/models.py`, then train using `main.py train`.

#### Dataset Augmentation
Besides using the built-in fake image generation functionality, you can:
1. Use public datasets (see links provided by the `download-info` command)
2. Create fake images using Photoshop or other image editing tools
3. Use AI generation tools (such as GANs) to create higher quality fake images

### Performance Optimization ⚡

- Using a GPU for model training and inference can significantly improve speed
- Increasing dataset size and diversity can improve model generalization ability
- Try different model architectures and hyperparameters for better performance
- In resource-limited environments, consider using smaller models like CNN or ResNet18

### Troubleshooting 🔨

#### Port Occupation Issue
On macOS, the AirPlay Receiver service may occupy the default port 5000. Solution:
1. Use another port (recommended):
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080
```

2. Or disable the AirPlay Receiver service in System Preferences:
System Preferences -> General -> AirDrop & Handoff -> Turn off AirPlay Receiver

#### Data Loading Issue
If you encounter a "dataset is empty" error, check:
1. Whether the data directory path is correct
2. Whether the directory contains supported image files (.jpg, .jpeg, .png)
3. Use the `generate_test_images.py` script to generate test data to verify the system

### Notes 📝

- Model performance is closely related to the training dataset and model selection
- The web application uses CPU for inference by default. For GPU acceleration, ensure CUDA is available in your environment
- On macOS, the AirPlay Receiver service may occupy port 5000, so using another port (such as 8080) is recommended
- When deploying in a production environment, it's advisable to use more secure file upload configurations and a more robust web server
- Detection results are for reference only and should not be used as the sole basis for judgment
- The system may have blind spots for certain types of image forgery techniques
- As forgery techniques continue to evolve, the system needs to be continuously updated to maintain effectiveness

### License 📄

This project is licensed under the MIT License

---

<a name="chinese-documentation"></a>
## 中文文档 🌏

### 概述 ℹ️
VerifyVision-Pro是一个基于深度学习的图像伪造检测系统，包括数据处理、模型训练和Web展示界面。

### 项目结构 📂

```
VerifyVision-Pro/
│
├── data/                      # 数据目录
│   ├── real/                  # 真实图像
│   ├── fake/                  # 伪造图像
│   └── processed/             # 预处理后的图像
│
├── models/                    # 模型目录
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

### 系统要求 🖥️

- Python 3.7+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装 📦

1. 克隆仓库
```bash
git clone https://github.com/lintsinghua/VerifyVision-Pro.git
cd VerifyVision-Pro
```

2. 创建虚拟环境（可选）
```bash
python -m venv imgvenv
source imgvenv/bin/activate  # Linux/Mac
imgvenv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始 🚀

项目提供了快速启动脚本，可以轻松体验完整功能：

1. **生成测试数据** 🎲
```bash
python generate_test_images.py
```
这会在data目录下生成20张真实图像和20张伪造图像，用于后续训练和测试。

2. **预处理图像** 🖌️
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

3. **训练模型** 🧠
```bash
python main.py train --real-dir data/processed/real --fake-dir data/processed/fake --model cnn --pretrained --epochs 5 --batch-size 4 --save-dir models/saved
```
注：可以使用较小的epochs值（如5）加速训练过程。

4. **启动Web应用** 🌐
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080 --debug
```
注：在macOS上默认的5000端口可能被AirPlay服务占用，建议使用8080端口。

5. **访问Web应用** 🖱️
打开浏览器访问 http://localhost:8080 即可使用系统。

### 数据准备 📊

#### 获取数据集 📥

可以使用以下方式获取数据：

1. **使用测试数据生成脚本**（推荐初次使用）：
```bash
python generate_test_images.py
```
这会自动创建真实和伪造图像，用于测试系统功能。

2. **使用公开数据集信息**：
```bash
python main.py download-info
```
程序会显示可用的公开图像伪造检测数据集链接，可手动下载。

3. **创建自己的数据集**：
- 收集真实图像到`data/real`目录
- 使用以下命令创建伪造图像：
```bash
python main.py create-fake --real-dir data/real --fake-dir data/fake --method splice --num-images 1000
```

#### 预处理数据 🔄

使用主程序接口处理图像：
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

参数说明：
- `--input-dir`：输入图像目录
- `--output-dir`：输出图像目录
- `--target-size`：目标图像大小，默认为224x224
- `--max-images`：最大处理图像数量，可选

### 模型训练 🏋️‍♂️

使用主程序接口训练模型：
```bash
python main.py train --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --pretrained --epochs 30 --batch-size 32 --save-dir models/saved
```

参数说明：
- `--real-dir`：真实图像目录
- `--fake-dir`：伪造图像目录
- `--model`：使用的模型，可选`efficientnet_b0`、`resnet18`、`resnet50`、`xception`、`cnn`
- `--pretrained`：是否使用预训练权重
- `--epochs`：训练周期数
- `--batch-size`：批次大小
- `--save-dir`：模型保存目录

更多参数请参考帮助信息：`python main.py train -h`

### 模型评估 📏

使用主程序接口评估模型性能：
```bash
python main.py evaluate --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --checkpoint models/saved/best_model.pth --results-dir results
```

参数说明：
- `--real-dir`：真实图像目录
- `--fake-dir`：伪造图像目录
- `--model`：使用的模型
- `--checkpoint`：模型检查点路径
- `--results-dir`：结果保存目录

### Web应用 🌐

使用主程序接口启动Web应用：
```bash
python main.py web --model-path models/saved/best_model.pth --model-name efficientnet_b0 --port 8080
```

参数说明：
- `--model-path`：模型路径
- `--model-name`：模型名称
- `--port`：端口号，推荐使用8080（macOS上5000端口可能被占用）
- `--debug`：添加此参数启用调试模式

#### Web应用使用方法 💻

1. 打开浏览器，访问 http://localhost:8080
2. 点击"选择文件"按钮，上传要检测的图像
3. 点击"上传并检测"按钮
4. 系统将显示检测结果，包括图像的真实或伪造判断以及相应的置信度

### 技术实现 🔧

本项目使用了以下主要技术：

1. **数据处理**：OpenCV, PIL, Albumentations
2. **深度学习框架**：PyTorch, TorchVision
3. **模型架构**：EfficientNet, ResNet, Xception
4. **Web框架**：Flask
5. **前端**：Bootstrap, JavaScript

### 高级用法 🔬

#### 自定义模型训练
可以通过修改 `src/models/models.py` 添加新的模型架构，然后使用 `main.py train` 进行训练。

#### 数据集扩充
除了使用内置的伪造图像生成功能，还可以：
1. 使用公开数据集（参见 `download-info` 命令提供的链接）
2. 使用Photoshop或其他图像编辑工具创建伪造图像
3. 使用AI生成工具（如GAN）创建更高质量的伪造图像

### 性能优化 ⚡

- 使用GPU进行模型训练和推理可显著提高速度
- 增加数据集规模和多样性可提高模型泛化能力
- 尝试不同的模型架构和超参数以获得更好的性能
- 在资源有限的环境中，可考虑使用较小的模型如CNN或ResNet18

### 常见问题解决 🔨

#### 端口占用问题
在macOS上，AirPlay Receiver服务可能会占用默认的5000端口。解决方案：
1. 使用其他端口（推荐）：
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080
```

2. 或在系统偏好设置中禁用AirPlay Receiver服务：
系统偏好设置 -> 通用 -> AirDrop和接力 -> 关闭AirPlay接收器

#### 数据加载问题
如果遇到"数据集为空"的错误，请检查：
1. 数据目录路径是否正确
2. 目录中是否包含支持的图像文件（.jpg, .jpeg, .png）
3. 使用`generate_test_images.py`脚本生成测试数据来验证系统

### 注意事项 📝

- 模型性能与训练数据集和模型选择密切相关
- Web应用默认使用CPU进行推理，如需GPU加速，请确保环境中有可用的CUDA
- 在macOS上，AirPlay Receiver服务可能占用5000端口，建议使用其他端口（如8080）
- 在生产环境中部署时，建议使用更安全的文件上传配置和更robust的Web服务器
- 检测结果仅供参考，不应作为唯一判断依据
- 系统对特定类型的图像伪造手法可能存在识别盲点
- 随着伪造技术的不断发展，系统需要持续更新以保持有效性

### 许可 📄

本项目使用 MIT 许可证 