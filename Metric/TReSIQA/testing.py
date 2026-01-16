

import os
import argparse
import random
import json
import numpy as np
import torch
from torchvision import transforms
from TReSIQA.models import TReS, Net
from PIL import Image

class config:
    seed = 2021
    version = 1
    patch_size = 224
    gpunum = '0'
    network = 'resnet50'
    nheadt = 16
    num_encoder_layerst = 2
    dim_feedforwardt = 64

seed = 721
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
solver = TReS(config, device, Net)
def evaluate_image(image_path): 
    process = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    image = process(image).unsqueeze(0).cuda()
    pred_score = solver.test(image)
    return pred_score.item()
