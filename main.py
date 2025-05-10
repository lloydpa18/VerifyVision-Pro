#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像伪造检测系统主入口
"""

import os
import argparse
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_utils.data_processor import preprocess_images, create_fake_dataset, download_datasets


def main():
    """主入口函数"""
    # 创建顶级解析器
    parser = argparse.ArgumentParser(description='图像伪造检测系统')
    subparsers = parser.add_subparsers(dest='command', help='子命令帮助')
    
    # 数据预处理子命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理图像')
    preprocess_parser.add_argument('--input-dir', type=str, required=True, help='输入图像目录')
    preprocess_parser.add_argument('--output-dir', type=str, required=True, help='输出图像目录')
    preprocess_parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224], help='目标图像大小')
    preprocess_parser.add_argument('--max-images', type=int, default=None, help='最大处理图像数量')
    
    # 创建伪造图像子命令
    fake_parser = subparsers.add_parser('create-fake', help='创建伪造图像')
    fake_parser.add_argument('--real-dir', type=str, required=True, help='真实图像目录')
    fake_parser.add_argument('--fake-dir', type=str, required=True, help='伪造图像输出目录')
    fake_parser.add_argument('--method', type=str, default='splice', choices=['copy', 'noise', 'color', 'splice'], help='伪造方法')
    fake_parser.add_argument('--num-images', type=int, default=1000, help='要创建的伪造图像数量')
    
    # 下载数据集子命令 - 无需参数
    subparsers.add_parser('download-info', help='获取数据集下载信息')
    
    # 训练模型子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--real-dir', type=str, required=True, help='真实图像目录')
    train_parser.add_argument('--fake-dir', type=str, required=True, help='伪造图像目录')
    train_parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    train_parser.add_argument('--num-workers', type=int, default=4, help='数据加载的工作线程数')
    train_parser.add_argument('--num-classes', type=int, default=2, help='分类数量')
    train_parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
    train_parser.add_argument('--lr-step', type=int, default=7, help='学习率衰减步长')
    train_parser.add_argument('--lr-gamma', type=float, default=0.1, help='学习率衰减系数')
    train_parser.add_argument('--save-dir', type=str, default='models/saved', help='模型保存目录')
    train_parser.add_argument('--save-interval', type=int, default=1, help='检查点保存间隔')
    train_parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    train_parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    train_parser.add_argument('--model', type=str, default='efficientnet_b0', 
                            choices=['efficientnet_b0', 'efficientnet_b2', 'resnet18', 'resnet50', 'xception', 'cnn'],
                            help='模型名称')
    train_parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器')
    train_parser.add_argument('--lr-scheduler', type=str, default='step', 
                            choices=['step', 'cosine', 'plateau', 'none'], help='学习率调度器')
    train_parser.add_argument('--epochs', type=int, default=30, help='训练周期数')
    train_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 评估模型子命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--real-dir', type=str, required=True, help='真实图像目录')
    eval_parser.add_argument('--fake-dir', type=str, required=True, help='伪造图像目录')
    eval_parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    eval_parser.add_argument('--num-workers', type=int, default=4, help='数据加载的工作线程数')
    eval_parser.add_argument('--num-classes', type=int, default=2, help='分类数量')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    eval_parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    eval_parser.add_argument('--results-dir', type=str, default='results', help='结果保存目录')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    eval_parser.add_argument('--model', type=str, default='efficientnet_b0', 
                            choices=['efficientnet_b0', 'efficientnet_b2', 'resnet18', 'resnet50', 'xception', 'cnn'],
                            help='模型名称')
    
    # 启动Web应用子命令
    web_parser = subparsers.add_parser('web', help='启动Web应用')
    web_parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    web_parser.add_argument('--model-name', type=str, default='efficientnet_b0', help='模型名称')
    web_parser.add_argument('--port', type=int, default=5000, help='端口号')
    web_parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据子命令执行相应的功能
    if args.command == 'preprocess':
        target_size = tuple(args.target_size)
        preprocess_images(args.input_dir, args.output_dir, target_size=target_size, max_images=args.max_images)
    
    elif args.command == 'create-fake':
        create_fake_dataset(args.real_dir, args.fake_dir, method=args.method, num_images=args.num_images)
    
    elif args.command == 'download-info':
        download_datasets()
    
    elif args.command == 'train':
        # 导入训练模块
        from src.training.train import train_model
        
        # 创建训练参数命名空间
        train_args = argparse.Namespace()
        for key, value in vars(args).items():
            if key != 'command':
                key_underscore = key.replace('-', '_')
                setattr(train_args, key_underscore, value)
        
        train_model(train_args)
    
    elif args.command == 'evaluate':
        # 导入评估模块
        from src.training.evaluate import evaluate_model
        
        # 创建评估参数命名空间
        eval_args = argparse.Namespace()
        for key, value in vars(args).items():
            if key != 'command':
                key_underscore = key.replace('-', '_')
                setattr(eval_args, key_underscore, value)
        
        evaluate_model(eval_args)
    
    elif args.command == 'web':
        from src.web.app import load_model, app
        
        # 加载模型
        load_model(args.model_path, args.model_name)
        
        # 启动应用
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 