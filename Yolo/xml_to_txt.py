import os
import xml.etree.ElementTree as ET
 
# 类别编号和名称的映射
class_mapping = {       # 替换为自己的类别编号和名称映射
    'person': '0',
    'bicycle': '1',
    'car': '2'
}
 
# 更新后的源文件夹和目标文件夹路径
source_folder = r'./LLVIP/annotations'      # 替换为需要转换为txt的xml文件的路径
target_folder = r'./LLVIP/labels'      # 替换为txt文件的保存路径
 
# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)
 
# 初始化图像宽度和高度变量
image_width = 0
image_height = 0
 
# 遍历源文件夹中的所有XML文件
for xml_file in os.listdir(source_folder):
    if xml_file.endswith('.xml'):
        # 构建完整的文件路径
        xml_path = os.path.join(source_folder, xml_file)
        
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            # 更新图像宽度和高度
            image_width = max(image_width, width)
            image_height = max(image_height, height)
        
        # 创建TXT文件名
        txt_file_name = xml_file.replace('.xml', '.txt')
        txt_file_path = os.path.join(target_folder, txt_file_name)
        
        # 写入TXT文件
        with open(txt_file_path, 'w') as txt_file:
            # 遍历所有对象
            for obj in root.findall('object'):
                # 获取类别编号
                name = obj.find('name').text
                category_id = class_mapping.get(name, None)
                if category_id is None:
                    continue  # 如果类别名称不在映射中，则跳过
                
                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 计算中心坐标 (x_center, y_center) 和宽高 (width, height)
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                
                # 归一化中心坐标和宽高
                normalized_x_center = x_center / image_width
                normalized_y_center = y_center / image_height
                normalized_width = width / image_width
                normalized_height = height / image_height
                
                # 写入对象信息，格式为类别编号 x_center y_center width height
                txt_file.write(f'{category_id} {normalized_x_center} {normalized_y_center} {normalized_width} {normalized_height}\n')
 
## 打印完成信息
print('Conversion from XML to TXT completed.')