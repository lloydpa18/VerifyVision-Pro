import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import get_model
from data_utils.dataset import get_dataloaders


def train_model(args):
    """
    训练图像伪造检测模型
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        args.real_dir, 
        args.fake_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # 获取模型
    model = get_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"未知的优化器: {args.optimizer}")
    
    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 训练循环
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n训练周期 {epoch+1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(dataloaders['train'], desc="训练"):
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(dataloaders['train'])
        train_acc = 100.0 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloaders['val'], desc="验证"):
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(dataloaders['val'])
        val_acc = 100.0 * val_correct / val_total
        
        # 更新学习率
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 打印统计信息
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 写入TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    # 训练完成，保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总时间: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像伪造检测训练')
    
    # 数据相关参数
    parser.add_argument('--real-dir', type=str, required=True, help='真实图像目录')
    parser.add_argument('--fake-dir', type=str, required=True, help='伪造图像目录')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--num-classes', type=int, default=2, help='分类数量')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        choices=['efficientnet_b0', 'efficientnet_b2', 'resnet18', 'resnet50', 'xception', 'cnn'],
                        help='模型名称')
    parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    
    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--lr-scheduler', type=str, default='step', 
                        choices=['step', 'cosine', 'plateau', 'none'], help='学习率调度器')
    parser.add_argument('--lr-step', type=int, default=10, help='学习率衰减步长')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='学习率衰减系数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--save-dir', type=str, default='models/saved', help='模型保存目录')
    parser.add_argument('--save-interval', type=int, default=5, help='检查点保存间隔')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args) 