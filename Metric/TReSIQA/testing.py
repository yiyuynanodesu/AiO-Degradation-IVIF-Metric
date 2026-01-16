

import os
import argparse
import random
import json
import numpy as np
import torch
from args import Configs
import logging
import data_loader
from torchvision import transforms
from models import TReS, Net
from PIL import Image

config = Configs()
print(config)
if torch.cuda.is_available():
    if len(config.gpunum)==1:
        device = torch.device("cuda", index=int(config.gpunum))
    else:
        device = torch.device("cpu")

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpunum

 # fix the seed if needed for reproducibility
if config.seed == 0:
    pass
else:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
solver = TReS(config, device, Net)
def evaluate_image(image): 
    pred_score = solver.test(image)
    return pred_score

if __name__ == '__main__':
    image_dir = './SameDegrad'
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        process = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
    		transforms.Normalize(mean=(0.485, 0.456, 0.406),
    							std=(0.229, 0.224, 0.225))
        ])
        image = Image.open(image_path).convert('RGB')
        image = process(image).unsqueeze(0).cuda()
        pred_score = evaluate_image(image)
        print(f'filename {filename} score: {pred_score}')
