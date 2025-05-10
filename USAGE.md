# 图像伪造检测系统使用指南

## 系统概述

本系统是基于深度学习的图像伪造检测技术研究与实现项目。系统能够检测图像是否经过伪造或篡改，适用于图像真实性鉴别的场景。

## 主要功能

1. **图像数据集预处理**：处理和准备用于训练模型的图像
2. **伪造图像生成**：创建不同类型的伪造图像用于训练
3. **模型训练**：训练深度学习模型识别真实和伪造图像
4. **模型评估**：评估模型性能，包括准确率、精确率、召回率等指标
5. **Web应用**：通过网页界面上传和检测图像
6. **测试数据生成**：通过脚本快速生成测试用真实和伪造图像

## 安装与配置

### 环境要求

- Python 3.7+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆或下载项目代码
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速使用指南

为了快速体验系统功能，我们推荐以下流程：

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

5. **访问Web应用**
打开浏览器访问 http://localhost:8080 即可使用系统。

## 详细使用方法

本系统提供了一个主入口脚本 `main.py`，支持多种操作命令：

### 1. 生成测试数据

项目提供了专门的脚本来生成测试用的真实和伪造图像：

```bash
python generate_test_images.py
```

该脚本会：
- 在data/real目录中生成20张随机真实图像
- 在data/fake目录中生成20张基于真实图像的伪造图像
- 自动创建必要的目录结构

### 2. 获取数据集信息

```bash
python main.py download-info
```

该命令会显示常用的图像伪造检测数据集下载链接，无需提供任何参数。

### 3. 预处理图像

```bash
python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224
```

参数说明：
- `--input-dir`：输入图像目录
- `--output-dir`：输出图像目录
- `--target-size`：目标图像大小，默认为224x224
- `--max-images`：最大处理图像数量，可选

### 4. 创建伪造图像

```bash
python main.py create-fake --real-dir data/real --fake-dir data/fake --method splice --num-images 1000
```

参数说明：
- `--real-dir`：真实图像目录
- `--fake-dir`：伪造图像输出目录
- `--method`：伪造方法，可选`copy`、`noise`、`color`、`splice`
- `--num-images`：要创建的伪造图像数量

### 5. 训练模型

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

### 6. 评估模型

```bash
python main.py evaluate --real-dir data/processed/real --fake-dir data/processed/fake --model efficientnet_b0 --checkpoint models/saved/best_model.pth --results-dir results
```

参数说明：
- `--real-dir`：真实图像目录
- `--fake-dir`：伪造图像目录
- `--model`：使用的模型
- `--checkpoint`：模型检查点路径
- `--results-dir`：结果保存目录

### 7. 启动Web应用

```bash
python main.py web --model-path models/saved/best_model.pth --model-name efficientnet_b0 --port 8080
```

参数说明：
- `--model-path`：模型路径
- `--model-name`：模型名称
- `--port`：端口号，推荐使用8080
  - 注意：在macOS上，默认的5000端口可能被AirPlay Receiver服务占用
- `--debug`：是否启用调试模式（添加此参数启用）

启动后，访问 http://localhost:8080 使用Web界面。

## Web应用使用

1. 打开浏览器，访问 http://localhost:8080
2. 点击"选择文件"按钮，上传要检测的图像
3. 点击"上传并检测"按钮
4. 系统将显示检测结果，包括图像的真实或伪造判断以及相应的置信度

## 高级用法

### 自定义模型训练

可以通过修改 `src/models/models.py` 添加新的模型架构，然后使用 `main.py train` 进行训练。

### 数据集扩充

除了使用内置的伪造图像生成功能，还可以：
1. 使用公开数据集（参见 `download-info` 命令提供的链接）
2. 使用Photoshop或其他图像编辑工具创建伪造图像
3. 使用AI生成工具（如GAN）创建更高质量的伪造图像

## 性能优化

- 使用GPU进行模型训练和推理可显著提高速度
- 增加数据集规模和多样性可提高模型泛化能力
- 尝试不同的模型架构和超参数以获得更好的性能
- 在资源有限的环境中，可考虑使用较小的模型如CNN或ResNet18

## 常见问题解决

### 端口占用问题

在macOS上，AirPlay Receiver服务可能会占用默认的5000端口。解决方案：
1. 使用其他端口（推荐）：
```bash
python main.py web --model-path models/saved/best_model.pth --model-name cnn --port 8080
```

2. 或在系统偏好设置中禁用AirPlay Receiver服务：
系统偏好设置 -> 通用 -> AirDrop和接力 -> 关闭AirPlay接收器

### 数据加载问题

如果遇到"数据集为空"的错误，请检查：
1. 数据目录路径是否正确
2. 目录中是否包含支持的图像文件（.jpg, .jpeg, .png）
3. 使用`generate_test_images.py`脚本生成测试数据来验证系统

## 注意事项

- 模型检测结果仅供参考，不应作为唯一判断依据
- 系统对特定类型的图像伪造手法可能存在识别盲点
- 随着伪造技术的不断发展，系统需要持续更新以保持有效性 