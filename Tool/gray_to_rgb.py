import os
from PIL import Image
import PIL
from natsort import natsorted
from tqdm import  tqdm

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    f_img = Image.open(f_name).convert('L')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

if __name__ == '__main__':
    fusion_folder = '' ## 融合图像所在的文件夹
    vi_filoder = '' ## 可见光图像所在的文件夹 两个文件夹里的图片命名需要保持一致
    file_list = os.listdir(fusion_folder)
    file_bar = tqdm(file_list)
    for file in natsorted(file_bar):
        f_name = os.path.join(fusion_folder, file)
        vi_name = os.path.join(vi_filoder, file)
        img2RGB(f_name, vi_name)
        file_bar.set_description('Y2RGB %s' % file)