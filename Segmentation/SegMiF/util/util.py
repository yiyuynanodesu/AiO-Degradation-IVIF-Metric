# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
import os

# FMB
# def get_palette():
#     unlabelled = [0,0,0]
#     car        = [64,0,128]
#     person     = [64,64,0]
#     bike       = [0,128,192]
#     curve      = [0,0,192]
#     car_stop   = [128,128,0]
#     guardrail  = [64,64,128]
#     color_cone = [192,128,128]
#     bump       = [192,64,0]
#     palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
#     return palette

def visualize(save_path, image_name, predictions):
    palette    = np.array([(0, 0, 0), (173, 229, 229), (187, 57, 134), (45, 163, 178), (206, 176, 47), (131, 54, 200), (56, 171, 83), (183, 71, 78), (66, 102, 167), (14, 127, 255), (138, 163, 91), (156, 98, 153), (101, 153, 140), (225, 214, 155), (136, 111, 89)])
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(save_path, image_name[i]))

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
