import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import get_model
from data_utils.dataset import get_dataloaders


def evaluate_model(args):
    """
    评估图像伪造检测模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        args.real_dir,
        args.fake_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 加载模型
    model = get_model(args.model, num_classes=args.num_classes)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 评估
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloaders['test'], desc="评估"):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取第二类（伪造）的概率
    
    test_loss = test_loss / len(dataloaders['test'])
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算准确率
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {accuracy:.2f}%")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_plot_labels = ['真实', '伪造']
    
    # 绘制并保存混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(cm_plot_labels))
    plt.xticks(tick_marks, cm_plot_labels, rotation=45)
    plt.yticks(tick_marks, cm_plot_labels)
    
    # 在每个单元格中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(args.results_dir, 'confusion_matrix.png'))
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=cm_plot_labels)
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告
    with open(os.path.join(args.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 绘制并保存ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.results_dir, 'roc_curve.png'))
    
    # 计算其他指标
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    # 保存结果摘要
    with open(os.path.join(args.results_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"测试损失: {test_loss:.4f}\n")
        f.write(f"测试准确率: {accuracy:.2f}%\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1_score:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
    
    print(f"\n评估结果已保存到 {args.results_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像伪造检测评估')
    
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
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    
    # 评估相关参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--results-dir', type=str, default='results', help='结果保存目录')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args) 