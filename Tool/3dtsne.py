import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import matplotlib.pylab as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model.Dino_Clip import DINOiser
from hydra import compose, initialize


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(seed=721)

def get_label(file):
    if "Haze" in file and "Rain" in file:
        return 0
    if "Haze" in file and "Low" in file:
        return 1
    if "Rain" in file:
        return 2
    if "Haze" in file:
        return 3
    if "exposure" in file:
        return 4
    if "light" in file:
        return 5   

class CustomDataset(Dataset):
    def __init__(self, vis_path):
        super().__init__()
        self.vis_path = vis_path
        self.filename_path = os.listdir(vis_path)
        self.process = transforms.Compose([
            transforms.Resize((224,224)),
        ])

    def get_frequency_domain_tensor(self, image):
        if image is None:
            raise ValueError("input image should be None")
        if len(image.shape) != 4 or image.shape[1] != 1:
            raise ValueError(f"should be [b, 1, h, w], but get {image.shape}")
    
        f = torch.fft.fft2(image, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))
        magnitude_spectrum = torch.abs(fshift)
    
        # Add a small epsilon to avoid log(0)
        magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)
        return magnitude_spectrum
    
    def get_rgb_frequency_domain_tensor(self, image):
        if image is None:
            raise ValueError("input image should be None")
        
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        
        (b, g, r) = image.split()
        b_tensor, g_tensor, r_tensor = to_tensor(b).unsqueeze(0), to_tensor(g).unsqueeze(0), to_tensor(r).unsqueeze(0)
    
        mag_image_b = self.get_frequency_domain_tensor(b_tensor)
        mag_image_g = self.get_frequency_domain_tensor(g_tensor)
        mag_image_r = self.get_frequency_domain_tensor(r_tensor)
        
        mag_image = torch.concat([mag_image_b, mag_image_g, mag_image_r],dim=1)
        return mag_image.squeeze(0)

    def __len__(self):
        return len(self.filename_path)

    def __getitem__(self, idx):
        filename = self.filename_path[idx]
        image_path = os.path.join(self.vis_path, filename)
        img = Image.open(image_path)
        img = self.process(img)
        freq = self.get_rgb_frequency_domain_tensor(img)
        label = get_label(filename)
        return freq, label

def plot_tsne3d(features, labels, class_text, save_path=None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # 指定3维，并初始化
    tsne = TSNE(n_components=3, 
                init='pca', 
                random_state=42,
                perplexity=min(30, features.shape[0]-1),  # 自适应perplexity
                max_iter=1000,
                learning_rate=200)
    
    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用t-SNE降维至3维
        print(f"t-SNE features shape: {tsne_features.shape}")
        print(f"t-SNE features range: [{tsne_features.min():.3f}, {tsne_features.max():.3f}]")
        
        # 检查t-SNE结果是否有效
        if np.any(np.isnan(tsne_features)):
            print("Error: t-SNE features contain NaN")
            return
        if np.all(tsne_features == 0):
            print("Error: All t-SNE features are zero")
            return
            
    except Exception as e:
        print(f"Error in t-SNE: {e}")
        return
    
    # 对数据进行归一化操作（添加小的epsilon避免除以0）
    x_min, x_max = np.min(tsne_features, axis=0), np.max(tsne_features, axis=0)
    if np.all((x_max - x_min) == 0):
        print("Warning: All features are identical, skipping normalization")
        embedded = tsne_features
    else:
        embedded = (tsne_features - x_min) / (x_max - x_min + 1e-8)
    
    hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db", "#dc8a78"]  # 粉红，暗红，浅绿，绿，蓝
    
    # 创建显示的figure - 使用更现代的API
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置3D视图角度
    ax.view_init(elev=20, azim=45)
    
    # 为每个类别绘制散点图
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label < len(hex):  # 确保索引在范围内
            mask = labels == label
            if np.sum(mask) > 0:
                ax.scatter(embedded[mask, 0], 
                          embedded[mask, 1], 
                          embedded[mask, 2],
                          c=hex[label],  # 使用颜色列表
                          marker="o",    # 使用实心圆点
                          s=30,          # 增大点的大小
                          alpha=0.8,     # 设置透明度
                          label=f'Class {label}')
    
    # 设置坐标轴标签
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title(f't-SNE 3D Visualization')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    
    # 设置图形布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # 重要：关闭图形释放内存
    print(f"Saved plot to: {save_path}")
    
    # 同时保存一个简化的2D视图用于快速检查
    fig2d = plt.figure(figsize=(10, 8))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE 2D View')
    plt.colorbar(label='Class')
    plt.tight_layout()
    save_name = "t-sne_2d.png"
    plt.savefig(save_name, format="png", dpi=200, bbox_inches='tight')
    plt.close(fig2d)
    print(f"Saved 2D plot to: {save_name}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_text = ['HazeRain', 'HazeLow', 'Rain', 'Haze', 'Exposure', 'LowLight']
    vis_path = '../dataset/Light_DDL-12/train/Visible'
    save_path = 'tsne_visualization.png'
    initialize(config_path="configs", version_base=None)            
    cfg = compose(config_name="clip_dinoiser.yaml")

    model_ex = DINOiser(cfg.model).to(device)
    model_ex.load_dino()
    model_ex.to(device)
    
    dataset = CustomDataset(vis_path)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            feature = model_ex(data)
            feature = feature.mean(dim=list(range(2, feature.dim())))
            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Test features shape: {features.shape}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    plot_tsne3d(features, labels, class_text, save_path=save_path)