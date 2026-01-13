import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from model.SPGFusion import SPGFusion
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(seed=721)

class CustomDataset(Dataset):
    def __init__(self, vis_path, ir_path):
        super().__init__()
        self.vis_path = vis_path
        self.ir_path = ir_path
        self.filename_path = os.listdir(vis_path)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filename_path)

    def __getitem__(self, idx):
        vis_filename = self.filename_path[idx]
        ir_filename = vis_filename
        vis_image = Image.open(os.path.join(self.vis_path, vis_filename))
        ir_image = Image.open(os.path.join(self.ir_path, ir_filename))
        vis_image = self.to_tensor(vis_image)
        ir_image = self.to_tensor(ir_image)
        label = get_label(vis_filename)
        return vis_image, ir_image, label

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_text = ['HazeRain', 'HazeLow', 'Rain', 'Haze', 'Exposure', 'LowLight']
    vis_path = '../dataset/LightDDL/train/Visible'
    ir_path = '../dataset/LightDDL/train/Infrared'
    n_components = 2
    perplexity = 50
    save_path = 'tsne_visualization.png'
    
    fusionNet = SPGFusion().to(device)
    fusion_weights = 'savePTH/epoch117/best_model.pth'
    ckpt = torch.load(fusion_weights, map_location='cpu')
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = fusionNet.load_state_dict(sd, strict=False)
    print('[LOAD] missing:', len(missing))
    print('[LOAD] unexpected:', len(unexpected))
    fusionNet.load_state_dict(sd, strict=True)
    fusionNet.eval()

    dataset = CustomDataset(vis_path, ir_path)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    # extract feature
    features = []
    labels = []
    
    with torch.no_grad():
        for vis_image, ir_image, label in data_loader:
            vis_image, ir_image, label = vis_image.to(device), ir_image.to(device), label.to(device)
            feature = fusionNet(vis_image,ir_image)
            features.append(feature.cpu().numpy())
            labels.append(label.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # TSNE
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity,
                max_iter=1000,
                random_state=721,
                init='pca',
                learning_rate='auto')
    
    features_tsne = tsne.fit_transform(features)

    # PLOT
    plt.figure(figsize=(4, 3))
    n_class = len(class_text)
    cmap = plt.cm.get_cmap('tab10', n_class)
    for i in range(n_class):
        mask = labels == i
        plt.scatter(features_tsne[mask, 0], 
                   features_tsne[mask, 1], 
                   c=[cmap(i)], 
                   label=class_text[i],
                   alpha=0.6,
                   s=n_class)
    
    plt.title('t-SNE with frequency', fontsize=16)
    plt.xlabel("t-SNE dimension 1", fontsize=14)
    plt.ylabel("t-SNE dimension 2", fontsize=14)
    plt.legend(title="Digits", fontsize=6, title_fontsize=6)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
