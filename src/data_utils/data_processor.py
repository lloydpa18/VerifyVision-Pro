import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random
import shutil


def download_datasets():
    """
    此函数提供获取常用图像伪造检测数据集的方法和链接
    """
    print("以下是常用的图像伪造检测数据集:")
    print("1. CASIA 2.0 数据集: https://github.com/namtpham/casia2groundtruth")
    print("2. Columbia Image Splicing Detection 数据集: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/")
    print("3. Coverage 数据集: https://github.com/wenbihan/coverage")
    print("4. NIST Nimble 2016 数据集: https://www.nist.gov/itl/iad/mig/nimble-2016-evaluation")
    print("5. PS-Battles 数据集: https://github.com/dbisUnibas/PS-Battles")
    
    print("\n您也可以使用自己的真实图像和通过工具如Photoshop或AI生成工具创建的伪造图像")
    print("如需下载，请访问上述链接手动下载数据集")


def preprocess_images(input_dir, output_dir, target_size=(256, 256), max_images=None):
    """
    将图像预处理并保存到输出目录
    
    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 输出图像目录
        target_size (tuple): 目标图像大小
        max_images (int): 最大处理图像数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    if max_images:
        random.shuffle(image_files)
        image_files = image_files[:max_images]
    
    print(f"正在处理 {len(image_files)} 张图像...")
    
    def process_image(img_file):
        try:
            img_path = os.path.join(input_dir, img_file)
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                # 尝试使用PIL读取
                pil_img = Image.open(img_path).convert('RGB')
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 调整大小
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # 保存处理后的图像
            out_path = os.path.join(output_dir, img_file)
            cv2.imwrite(out_path, img_resized)
            
            return True
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
            return False
    
    # 使用线程池并行处理图像
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_image, image_files), total=len(image_files)))
    
    success_count = results.count(True)
    print(f"成功处理 {success_count}/{len(image_files)} 张图像")
    return success_count


def create_fake_dataset(real_dir, fake_dir, method='copy', num_images=1000):
    """
    创建伪造图像数据集
    
    Args:
        real_dir (str): 真实图像目录
        fake_dir (str): 伪造图像输出目录
        method (str): 伪造方法 ('copy', 'noise', 'color', 'splice')
        num_images (int): 要创建的伪造图像数量
    """
    os.makedirs(fake_dir, exist_ok=True)
    
    # 获取所有真实图像
    real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(real_images) == 0:
        print(f"在 {real_dir} 中没有找到图像")
        return 0
    
    # 确保不超过可用的真实图像数量
    num_images = min(num_images, len(real_images))
    
    # 随机选择图像
    selected_images = random.sample(real_images, num_images)
    
    print(f"正在创建 {num_images} 张伪造图像，使用 {method} 方法...")
    
    def process_image(img_file):
        try:
            img_path = os.path.join(real_dir, img_file)
            fake_img_path = os.path.join(fake_dir, f"fake_{img_file}")
            
            # 读取图像
            img = cv2.imread(img_path)
            
            if method == 'copy':
                # 简单复制并重命名
                shutil.copy(img_path, fake_img_path)
            
            elif method == 'noise':
                # 添加随机噪声
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                noisy_img = cv2.add(img, noise)
                cv2.imwrite(fake_img_path, noisy_img)
            
            elif method == 'color':
                # 颜色调整
                hue_shift = random.randint(-10, 10)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
                modified = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(fake_img_path, modified)
            
            elif method == 'splice':
                # 图像拼接 - 随机选择另一张图像进行拼接
                other_img_file = random.choice([f for f in real_images if f != img_file])
                other_img_path = os.path.join(real_dir, other_img_file)
                other_img = cv2.imread(other_img_path)
                
                # 调整大小以匹配原图像
                other_img = cv2.resize(other_img, (img.shape[1], img.shape[0]))
                
                # 在随机位置创建拼接
                h, w = img.shape[:2]
                x = random.randint(0, w//2)
                y = random.randint(0, h//2)
                width = random.randint(w//4, w//2)
                height = random.randint(h//4, h//2)
                
                # 将区域从other_img复制到img
                img[y:y+height, x:x+width] = other_img[y:y+height, x:x+width]
                cv2.imwrite(fake_img_path, img)
            
            return True
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
            return False
    
    # 使用线程池并行处理图像
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_image, selected_images), total=len(selected_images)))
    
    success_count = results.count(True)
    print(f"成功创建 {success_count}/{len(selected_images)} 张伪造图像")
    return success_count


def split_dataset(data_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        data_dir (str): 数据目录
        output_dir (str): 输出目录
        split_ratio (tuple): 训练、验证和测试集的比例
    """
    # 确保比例总和为1
    assert sum(split_ratio) == 1.0, "比例总和必须为1"
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有图像
    images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    # 计算分割索引
    n_train = int(len(images) * split_ratio[0])
    n_val = int(len(images) * split_ratio[1])
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    # 复制文件
    for img in train_images:
        shutil.copy(os.path.join(data_dir, img), os.path.join(train_dir, img))
    
    for img in val_images:
        shutil.copy(os.path.join(data_dir, img), os.path.join(val_dir, img))
    
    for img in test_images:
        shutil.copy(os.path.join(data_dir, img), os.path.join(test_dir, img))
    
    print(f"数据集分割完成:")
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")
    print(f"测试集: {len(test_images)} 张图像")


if __name__ == "__main__":
    # 示例用法
    print("数据处理工具")
    print("使用方法:")
    print("1. 从 download_datasets() 函数获取数据集下载链接")
    print("2. 使用 preprocess_images() 处理图像")
    print("3. 使用 create_fake_dataset() 创建伪造图像")
    print("4. 使用 split_dataset() 分割数据集") 