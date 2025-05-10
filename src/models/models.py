import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EfficientNetForensics(nn.Module):
    """基于EfficientNet的图像伪造检测模型"""
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=2):
        """
        初始化模型
        
        Args:
            model_name (str): EfficientNet型号
            pretrained (bool): 是否使用预训练权重
            num_classes (int): 分类数量
        """
        super(EfficientNetForensics, self).__init__()
        
        # 加载预训练的EfficientNet模型
        if pretrained:
            self.model = getattr(models, model_name)(weights='IMAGENET1K_V1')
        else:
            self.model = getattr(models, model_name)(weights=None)
        
        # 获取特征提取器（除了最后的全连接层）
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        # 获取特征维度
        if 'efficientnet_b0' in model_name:
            num_features = 1280
        elif 'efficientnet_b1' in model_name:
            num_features = 1280
        elif 'efficientnet_b2' in model_name:
            num_features = 1408
        elif 'efficientnet_b3' in model_name:
            num_features = 1536
        else:
            num_features = 1792  # b4 and higher
        
        # 添加自定义分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """前向传播"""
        # 使用特征提取器获取特征
        features = self.features(x)
        # 转为一维以用于全连接层
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        # 通过分类器获取输出
        output = self.classifier(features)
        return output


class ResNetForensics(nn.Module):
    """基于ResNet的图像伪造检测模型"""
    
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=2):
        """
        初始化模型
        
        Args:
            model_name (str): ResNet型号
            pretrained (bool): 是否使用预训练权重
            num_classes (int): 分类数量
        """
        super(ResNetForensics, self).__init__()
        
        # 加载预训练的ResNet模型
        if pretrained:
            self.model = getattr(models, model_name)(weights='IMAGENET1K_V1')
        else:
            self.model = getattr(models, model_name)(weights=None)
        
        # 获取特征提取器
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        # 获取特征维度
        if model_name == 'resnet18' or model_name == 'resnet34':
            num_features = 512
        else:
            num_features = 2048
        
        # 添加自定义分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.features(x)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output


class XceptionForensics(nn.Module):
    """
    基于Xception的图像伪造检测模型
    Xception架构在图像伪造检测任务中表现出色
    """
    
    def __init__(self, num_classes=2):
        super(XceptionForensics, self).__init__()
        # 注意：需要安装pretrainedmodels库或实现自定义的Xception
        # 这里我们使用一个简化的实现
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 进入深度可分离卷积
        # 简化版本，实际应该有多个深度可分离卷积块
        self.block1 = self._make_block(64, 128, 2)
        self.block2 = self._make_block(128, 256, 2)
        self.block3 = self._make_block(256, 728, 2)
        
        # 中间流
        self.block4 = self._make_block(728, 728, 1)
        self.block5 = self._make_block(728, 728, 1)
        self.block6 = self._make_block(728, 728, 1)
        
        # 退出流
        self.block7 = self._make_block(728, 1024, 2)
        
        self.conv3 = nn.Conv2d(1024, 1536, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(1536, 2048, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_block(self, in_channels, out_channels, stride):
        """创建深度可分离卷积块"""
        return nn.Sequential(
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 深度卷积
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 逐点卷积
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # 入口流
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # 深度可分离卷积块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 中间流
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        # 退出流
        x = self.block7(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class CNNForensics(nn.Module):
    """自定义CNN结构的图像伪造检测模型"""
    
    def __init__(self, num_classes=2):
        super(CNNForensics, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第五个卷积块
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        # 假设输入图像为224x224，经过5次下采样后为7x7
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # 卷积块4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # 卷积块5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        
        return x


def get_model(model_name, num_classes=2, pretrained=True):
    """
    获取指定的模型
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 分类数量
        pretrained (bool): 是否使用预训练权重
        
    Returns:
        nn.Module: 模型实例
    """
    if model_name.startswith('efficientnet'):
        return EfficientNetForensics(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name.startswith('resnet'):
        return ResNetForensics(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'xception':
        return XceptionForensics(num_classes=num_classes)
    elif model_name == 'cnn':
        return CNNForensics(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型名称: {model_name}") 