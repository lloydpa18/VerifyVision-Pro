import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageForensicsDataset(Dataset):
    """图像伪造检测数据集类"""
    
    def __init__(self, real_dir, fake_dir, transform=None, split='train'):
        """
        初始化数据集
        
        Args:
            real_dir (str): 真实图像目录路径
            fake_dir (str): 伪造图像目录路径
            transform: 图像变换
            split (str): 数据集分割类型 ('train', 'val', 'test')
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.split = split
        
        # 获取所有图像文件路径
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) 
                           if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) 
                           if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 设置随机种子以确保可重复性
        random.seed(42)
        
        # 打乱数据
        random.shuffle(self.real_images)
        random.shuffle(self.fake_images)
        
        # 根据分割类型选择数据
        if split == 'train':
            self.real_images = self.real_images[:int(0.7 * len(self.real_images))]
            self.fake_images = self.fake_images[:int(0.7 * len(self.fake_images))]
        elif split == 'val':
            self.real_images = self.real_images[int(0.7 * len(self.real_images)):int(0.85 * len(self.real_images))]
            self.fake_images = self.fake_images[int(0.7 * len(self.fake_images)):int(0.85 * len(self.fake_images))]
        elif split == 'test':
            self.real_images = self.real_images[int(0.85 * len(self.real_images)):]
            self.fake_images = self.fake_images[int(0.85 * len(self.fake_images)):]
        
        # 合并所有图像路径并创建标签
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0为真实，1为伪造
        
        # 同时打乱图像和标签
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取数据集中的一项"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # 如果使用albumentations库
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # 如果使用torchvision的transforms
                image = self.transform(image)
        
        return image, label

def get_transforms(img_size=224):
    """
    获取训练和验证用的图像变换
    
    Args:
        img_size (int): 图像大小
        
    Returns:
        dict: 包含训练和验证变换的字典
    """
    # 使用Albumentations进行数据增强
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return {'train': train_transform, 'val': val_transform, 'test': val_transform}

def get_dataloaders(real_dir, fake_dir, batch_size=32, img_size=224, num_workers=4):
    """
    获取数据加载器
    
    Args:
        real_dir (str): 真实图像目录
        fake_dir (str): 伪造图像目录
        batch_size (int): 批次大小
        img_size (int): 图像大小
        num_workers (int): 数据加载线程数
        
    Returns:
        dict: 包含训练、验证和测试数据加载器的字典
    """
    transforms_dict = get_transforms(img_size)
    
    # 创建数据集
    train_dataset = ImageForensicsDataset(real_dir, fake_dir, transforms_dict['train'], 'train')
    val_dataset = ImageForensicsDataset(real_dir, fake_dir, transforms_dict['val'], 'val')
    test_dataset = ImageForensicsDataset(real_dir, fake_dir, transforms_dict['test'], 'test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 