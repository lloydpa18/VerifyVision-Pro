#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成测试图像，用于项目测试
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# 创建目录
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 生成真实图像
def generate_real_images(output_dir, num_images=10, size=(256, 256)):
    ensure_dir(output_dir)
    
    for i in range(num_images):
        # 创建一个随机背景颜色的图像
        bg_color = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        )
        
        img = Image.new('RGB', size, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制一些随机形状
        for _ in range(random.randint(3, 6)):
            # 随机颜色
            color = (
                random.randint(0, 180),
                random.randint(0, 180),
                random.randint(0, 180)
            )
            
            # 随机选择形状类型（圆形、矩形、椭圆）
            shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
            
            # 随机位置和大小，确保范围有效
            max_width = size[0] - 1
            max_height = size[1] - 1
            
            x1 = random.randint(0, max_width - 40)  # 至少留出40像素的空间
            y1 = random.randint(0, max_height - 40)
            
            # 确保x2和y2的范围有效
            x2_min = min(x1 + 20, max_width - 1)
            x2_max = min(x1 + 100, max_width)
            y2_min = min(y1 + 20, max_height - 1)
            y2_max = min(y1 + 100, max_height)
            
            # 如果范围无效，则调整
            if x2_min >= x2_max:
                x2_min = x1 + 10
                x2_max = x1 + 20
            
            if y2_min >= y2_max:
                y2_min = y1 + 10
                y2_max = y1 + 20
                
            x2 = random.randint(x2_min, x2_max)
            y2 = random.randint(y2_min, y2_max)
            
            # 绘制形状
            if shape_type == 'circle':
                radius = min(x2 - x1, y2 - y1) // 2
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=color)
            elif shape_type == 'rectangle':
                draw.rectangle((x1, y1, x2, y2), fill=color)
            elif shape_type == 'ellipse':
                draw.ellipse((x1, y1, x2, y2), fill=color)
        
        # 保存图像
        file_path = os.path.join(output_dir, f'real_{i+1:04d}.jpg')
        img.save(file_path, 'JPEG', quality=95)
        
        print(f'生成真实图像: {file_path}')
    
    return num_images

# 使用一个图像创建伪造图像（简单的拼接操作）
def create_fake_from_real(real_image_path, output_path):
    # 读取真实图像
    img = Image.open(real_image_path)
    width, height = img.size
    
    # 创建伪造部分（一个明显的矩形）
    draw = ImageDraw.Draw(img)
    
    # 随机位置，确保范围有效
    x1 = random.randint(width // 4, width // 2)
    y1 = random.randint(height // 4, height // 2)
    
    # 确保x2和y2的范围有效
    x2_min = min(x1 + 30, width - 20)
    x2_max = min(x1 + 100, width - 10)
    y2_min = min(y1 + 30, height - 20)
    y2_max = min(y1 + 100, height - 10)
    
    # 如果范围无效，则调整
    if x2_min >= x2_max:
        x2_min = x1 + 10
        x2_max = x1 + 20
    
    if y2_min >= y2_max:
        y2_min = y1 + 10
        y2_max = y1 + 20
    
    x2 = random.randint(x2_min, x2_max)
    y2 = random.randint(y2_min, y2_max)
    
    # 绘制明显的矩形（伪造部分）
    fake_color = (255, 0, 0)  # 红色
    draw.rectangle((x1, y1, x2, y2), fill=fake_color)
    
    # 添加文本标签，表明这是伪造图像
    try:
        draw.text((10, 10), "Fake", fill=(255, 255, 255))
    except:
        # 如果缺少字体，简单地绘制一个标记
        draw.rectangle((5, 5, 30, 20), fill=(255, 255, 255), outline=(0, 0, 0))
    
    # 保存伪造图像
    img.save(output_path, 'JPEG', quality=95)
    
    print(f'生成伪造图像: {output_path}')

# 生成伪造图像
def generate_fake_images(real_dir, fake_dir, num_images=10):
    ensure_dir(fake_dir)
    
    # 获取真实图像列表
    real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(real_images) == 0:
        print(f"在 {real_dir} 中没有找到图像")
        return 0
    
    # 确保不超过可用的真实图像数量
    num_images = min(num_images, len(real_images))
    
    # 随机选择图像
    selected_images = random.sample(real_images, num_images)
    
    for i, img_file in enumerate(selected_images):
        img_path = os.path.join(real_dir, img_file)
        fake_img_path = os.path.join(fake_dir, f'fake_{i+1:04d}.jpg')
        
        create_fake_from_real(img_path, fake_img_path)
    
    return num_images

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    random.seed(42)
    
    # 设置输出目录
    real_dir = 'data/real'
    fake_dir = 'data/fake'
    
    # 生成真实和伪造图像
    num_real = generate_real_images(real_dir, num_images=20)
    num_fake = generate_fake_images(real_dir, fake_dir, num_images=20)
    
    print(f"已生成 {num_real} 张真实图像和 {num_fake} 张伪造图像")
    
    # 为处理后的图像创建目录
    ensure_dir('data/processed/real')
    ensure_dir('data/processed/fake')
    
    print("现在您可以使用以下命令处理图像:")
    print("python main.py preprocess --input-dir data/real --output-dir data/processed/real --target-size 224 224")
    print("python main.py preprocess --input-dir data/fake --output-dir data/processed/fake --target-size 224 224") 