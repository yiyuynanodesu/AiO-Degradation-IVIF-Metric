# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, result_path=None,label_path=None):
        super(Fusion_dataset, self).__init__()
        data_dir_result = result_path
        data_dir_label = label_path
        self.filepath_result, self.filenames_result = prepare_data_path(data_dir_result)
        self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
        self.length = len(self.filenames_result)

    def __getitem__(self, index):
        result_name = self.filenames_result[index]
        result_path = self.filepath_result[index]
        label_path = self.filepath_label[index]
        image_result = np.array(Image.open(result_path))
        image_result = (
            np.asarray(Image.fromarray(image_result), dtype=np.float32).transpose(
                (2, 0, 1)
            )
            / 255.0
        )
        label = np.array(Image.open(label_path))
        label = np.asarray(Image.fromarray(label), dtype=np.int64)
    
        return (
            torch.tensor(image_result),
            torch.tensor(label),
            result_name,
        )

    def __len__(self):
        return self.length