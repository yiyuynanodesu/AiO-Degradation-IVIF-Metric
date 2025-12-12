# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np

import plotly.graph_objects as go
import plotly.offline as pyo

config = 'configs/clipiqa/clipiqa_attribute_test.py'
checkpoint = None
device = 0

model = init_model(
    config, checkpoint, device=torch.device('cuda', device))

def evaluate_image(image_path):
    output, attributes = restoration_inference(model, os.path.join(image_path), return_attributes=True)
    output = output.float().detach().cpu().numpy()
    attributes = attributes.float().detach().cpu().numpy()[0]
    return np.sum(attributes) / len(attributes)