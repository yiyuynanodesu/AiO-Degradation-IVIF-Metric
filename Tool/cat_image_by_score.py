import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from curriculum_learning.entropy2dSpeedUp import calcEntropy2dSpeedUp

img_dir = './dataset/MSRS/test/vi'
img_score_dict = {}

# 计算所有图像的分数
for filename in os.listdir(img_dir):
    image_path = os.path.join(img_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue  # 跳过无法读取的图像
        
    image_ = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    
    # 采用信息熵+饱和度+明度准则
    image_gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
    image_entropy = calcEntropy2dSpeedUp(image_gray, 3, 3)
    image_hsv = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(image_hsv)
    
    # 明度（V）
    v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
    average_v = sum(v) / len(v) if len(v) > 0 else 0
    
    # 饱和度（S）
    s = S.ravel()[np.flatnonzero(S)]
    average_s = sum(s) / len(s) if len(s) > 0 else 0
    
    score = image_entropy**2 * np.math.log(1 + average_v * average_s, np.math.e) / 1000
    img_score_dict[filename] = score

# 按照分数从大到小排序
sorted_scores = sorted(img_score_dict.items(), key=lambda x: x[1], reverse=True)

# 显示前20张图像
plt.figure(figsize=(20, 16))
for i, (filename, score) in enumerate(sorted_scores[:20]):
    image_path = os.path.join(img_dir, filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式用于显示
    
    plt.subplot(4, 5, i + 1)  # 4行5列的子图
    plt.imshow(image_rgb)
    plt.title(f'{filename}\nScore: {score:.4f}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# 打印前20个图像的分数排名
print("前20张图像的分数排名：")
for i, (filename, score) in enumerate(sorted_scores[:20], 1):
    print(f'{i:2d}. {filename}: {score:.4f}')