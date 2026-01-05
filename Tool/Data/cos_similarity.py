# -*- coding: utf-8 -*-
# !/usr/bin/env python
# 余弦相似度计算
from PIL import Image
import os
from numpy import average, dot, linalg
import numpy as np
# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=True):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

dino_file_path = './dino_file'
dino_paths = os.listdir(dino_file_path)

for dino_filename_1 in dino_paths:
    # score = []
    for dino_filename_2 in dino_paths:
        image1 = Image.open(os.path.join(dino_file_path, dino_filename_1))
        image2 = Image.open(os.path.join(dino_file_path, dino_filename_2))
        cosin = image_similarity_vectors_via_numpy(image1, image2)
        print(f'{dino_filename_1} and {dino_filename_2} cos similarity is {cosin}')