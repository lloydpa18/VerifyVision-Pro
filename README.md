# 基于深度学习的图像伪造检测系统

本项目实现了一个基于深度学习的图像伪造检测系统，包括数据处理、模型训练和Web展示界面。

## 项目结构

```
imgproject/
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
├── USAGE.md                   # 使用指南
└── README.md                  # 项目说明
```

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/imgproject.git
cd imgproject
```

2. 创建虚拟环境（可选）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

项目提供了快速启动脚本，可以轻松体验完整功能：

1. **生成测试数据**
```bash
python generate_test_images.py
```
这会在data目录下生成20张真实图像和20张伪造图像，用于后续训练和测试。

2. **预处理图像**
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

3. **训练模型**
```bash
python main.py train --real-dir data/processed/real --fake-dir data/processed/fake --model cnn --pretrained --epochs 5 --batch-size 4 --save-dir models/saved
```
注：可以使用较小的epochs值（如5）加速训练过程。

4. **启动Web应用**
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080 --debug
```
注：在macOS上默认的5000端口可能被AirPlay服务占用，建议使用8080端口。

完成上述步骤后，打开浏览器访问 http://localhost:8080 即可使用系统。

## 数据准备

### 获取数据集

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

### 预处理数据

使用主程序接口处理图像：
```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224
```

## 模型训练

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

## 模型评估

使用主程序接口评估模型性能：
```bash
python main.py evaluate --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --checkpoint models/saved/best_model.pth --results-dir results
```

参数说明：
- `--checkpoint`：模型检查点路径
- `--results-dir`：结果保存目录

## Web应用

使用主程序接口启动Web应用：
```bash
python main.py web --model-path models/saved/best_model.pth --model-name efficientnet_b0 --port 8080
```

参数说明：
- `--model-path`：模型路径
- `--model-name`：模型名称
- `--port`：端口号，推荐使用8080（macOS上5000端口可能被占用）
- `--debug`：添加此参数启用调试模式

访问 http://localhost:8080 使用图像伪造检测系统。

## 技术实现

本项目使用了以下主要技术：

1. **数据处理**：OpenCV, PIL, Albumentations
2. **深度学习框架**：PyTorch, TorchVision
3. **模型架构**：EfficientNet, ResNet, Xception
4. **Web框架**：Flask
5. **前端**：Bootstrap, JavaScript

## 注意事项

- 模型性能与训练数据集和模型选择密切相关
- Web应用默认使用CPU进行推理，如需GPU加速，请确保环境中有可用的CUDA
- 在macOS上，AirPlay Receiver服务可能占用5000端口，建议使用其他端口（如8080）
- 在生产环境中部署时，建议使用更安全的文件上传配置和更robust的Web服务器

## 许可

本项目使用 MIT 许可证 