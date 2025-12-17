import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model.Adapter_one import Adapter as one
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
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filename_path)

    def __getitem__(self, idx):
        filename = self.filename_path[idx]
        image_path = os.path.join(self.vis_path, filename)
        img = Image.open(image_path)
        img = self.process(img)
        label = get_label(filename)
        return img, label

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_text = ['HazeRain', 'HazeLow', 'Rain', 'Haze', 'LowLight', 'Exposure']
    adapter_weight = './pretrained_weights/one.pth'
    vis_path = '../dataset/Light_DDL-12/train/Visible'
    n_components = 2
    perplexity = 50
    save_path = 'tsne_visualization.png'
    
    cls_model = one()
    cls_model.load_state_dict(torch.load(adapter_weight), strict=False)
    cls_model.to(device)
    cls_model.eval()
    dataset = CustomDataset(vis_path)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    # extract feature
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            feature = cls_model(data)
            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
            
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
