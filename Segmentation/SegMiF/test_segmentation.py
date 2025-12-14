# coding:utf-8
import torch
import os
import argparse
import time
import numpy as np

from core.model_fusion import Network3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TaskFusion_dataset2 import Fusion_dataset

from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import os, argparse, time, datetime, sys, shutil, stat
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from model_fusion_seg_tzy4 import Network
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat

def val_segformer2(args, model, n_class):
    file = './val_seg_'+args.model_name+'.txt'
    file_o = open(file,'a+')
    conf_total = np.zeros((n_class, n_class))

    class_dict = {}
    for i in range(class_text):
        class_dict[i] = class_text[i]

    score_thres = 0.7
    ignore_idx = 255
    h=480
    w=640

    image_size=(h,w)
    n_min = 8 * 640 * 480 // 8

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)

    test_dataset = Fusion_dataset('val', result_path=args.result_path, label_path= args.label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_result,label,name) in enumerate(test_loader):
            images_result = Variable(images_result)
            if args.gpu >= 0:
                images_result = images_result.to(device)

            logits,_, seg1, = model.forward(images_result)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf


        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n############################Class Metric####################################')
        print("*precision_per_class: \n")
        for i in range(len(precision_per_class)):
            print(f'{class_dict[i]} : Precision: {precision_per_class[i]} Recall: {recall_per_class[i]} IOU: {iou_per_class[i]} ')
        print('\n################################Average####################################')
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        return iou_per_class.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='model')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--result_path', '-j', type=int, default='')
    parser.add_argument('--label_path', '-j', type=int, default='')
    args = parser.parse_args()
    class_text = []
    seg_model_path = './pretrained/model-fusion_add_final2.pth'
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    model_seg = Network3('mit_b3', len(class_text)).cuda()
    model_seg.load_state_dict(torch.load(seg_model_path))
    val_segformer2(args, model_seg, class_text)
