import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser(description='Convert LLVIP XML annotations to YOLOv5 format')
    parser.add_argument('--annotation_path', type=str,
                        default='/home/dataset/LLVIP/Annotations',
                        help='folder containing xml files')
    parser.add_argument('--image_path', type=str, default='/home/dataset/LLVIP/infrared/train',
                        help='image path, e.g. /root/LLVIP/infrared/train')
    parser.add_argument('--txt_save_path', type=str, default='/home/dataset/LLVIP/labels/train',
                        help='txt path containing txt files in yolov5 format')
    opt = parser.parse_args()
    return opt


opt = parse_opt()


def convert_LLVIP_annotation(anno_path, image_path, txt_path):
    # 创建保存txt文件的目录
    os.makedirs(txt_path, exist_ok=True)

    # 获取所有图像文件名
    image_files = os.listdir(image_path)

    for i in tqdm(image_files):
        try:
            # 获取图像文件名(不含扩展名)
            img_name = i.split(".")[0]

            # 构建对应的XML文件路径
            xml_file = os.path.join(anno_path, img_name + '.xml')

            # 检查XML文件是否存在
            if not os.path.exists(xml_file):
                print(f"警告: {xml_file} 不存在，跳过")
                continue

            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取所有对象
            objects = root.findall('object')

            # 如果没有对象，创建空的txt文件
            if not objects:
                with open(os.path.join(txt_path, img_name + '.txt'), 'w') as f:
                    pass
                continue

            # 处理每个对象
            annotations = []
            for obj in objects:
                try:
                    # 获取边界框坐标
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text.strip())
                    xmax = int(bbox.find('xmax').text.strip())
                    ymin = int(bbox.find('ymin').text.strip())
                    ymax = int(bbox.find('ymax').text.strip())

                    # 转换为YOLO格式
                    x_center = (0.5 * (xmin + xmax)) / 1280
                    y_center = (0.5 * (ymin + ymax)) / 1024
                    width = (xmax - xmin) / 1280
                    height = (ymax - ymin) / 1024

                    # 构建标注行
                    annotation = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    annotations.append(annotation)
                except Exception as e:
                    print(f"警告: 处理 {xml_file} 中的对象时出错: {e}")
                    continue

            # 将所有标注写入txt文件
            if annotations:
                with open(os.path.join(txt_path, img_name + '.txt'), 'w') as f:
                    f.write('\n'.join(annotations) + '\n')

        except Exception as e:
            print(f"错误: 处理文件 {i} 时出错: {e}")
            continue


anno_path = opt.annotation_path
image_path = opt.image_path
txt_path = opt.txt_save_path
convert_LLVIP_annotation(anno_path, image_path, txt_path)