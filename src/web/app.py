import os
import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import get_model

# 创建Flask应用
app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
app.secret_key = 'image_forensics_detector_key'

# 配置上传文件目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上传大小

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 全局变量
model = None
device = None
img_size = 224
transform = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path, model_name='efficientnet_b0'):
    """
    加载模型
    
    Args:
        model_path (str): 模型路径
        model_name (str): 模型名称
        
    Returns:
        模型和设备
    """
    global model, device, transform
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_model(model_name, num_classes=2)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 创建图像变换
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print(f"模型 '{model_name}' 已加载到 {device}")


def predict_image(image_path):
    """
    预测图像
    
    Args:
        image_path (str): 图像路径
        
    Returns:
        预测结果和概率
    """
    # 加载并处理图像
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # 应用变换
    img_tensor = transform(image=img_np)['image']
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
    
    # 返回结果
    return {
        'class': '伪造' if pred_class == 1 else '真实',
        'fake_prob': float(probs[1].item()) * 100,  # 伪造概率
        'real_prob': float(probs[0].item()) * 100   # 真实概率
    }


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        flash('未选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测图像
        result = predict_image(filepath)
        
        # 添加文件路径到结果
        result['image_path'] = 'uploads/' + filename
        
        return render_template('result.html', result=result)
    
    flash('不支持的文件类型，请上传JPG、JPEG或PNG格式的图像')
    return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API端点用于预测"""
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测图像
        result = predict_image(filepath)
        
        # 添加文件路径到结果
        result['image_path'] = 'uploads/' + filename
        
        return jsonify(result)
    
    return jsonify({'error': '不支持的文件类型，请上传JPG、JPEG或PNG格式的图像'}), 400


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='图像伪造检测Web应用')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--model-name', type=str, default='efficientnet_b0', help='模型名称')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.model_name)
    
    # 启动应用
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 