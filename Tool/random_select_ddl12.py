import os
import shutil
import random

'''
随机从DDL-12 雨 雾 低光 高曝光 雨+雾 雾+低光 每个退化 的 每个程度 中随机选择一定数量的图片 重新组成新的数据集
'''

random.seed(721)

target_dataset_path = './dataset/Light_DDL-12'

# 将所有图片都以png形式保留
save_ext = 'png'
os.makedirs(target_dataset_path, exist_ok=True)

def move_single():
    dataset_path = './dataset/DDL-12/Single'
    '''
    对于单个退化 从每个程度的训练集中选250张 测试集选45张 这样单个退化有1000张训练数据 180张测试数据
    '''
    
    select_num = {
        'train': 250,  # 训练集选择250张
        'test': 45     # 测试集选择45张
    }
    
    # 遍历每种退化类型
    for dtype in os.listdir(dataset_path):
        dataset_dtype_path = os.path.join(dataset_path, dtype)
        
        # 遍历每个退化程度
        for degree in os.listdir(dataset_dtype_path):
            dataset_dtype_degree_path = os.path.join(dataset_dtype_path, degree)
    
            # 遍历训练集和测试集
            for phrase in os.listdir(dataset_dtype_degree_path):
                dataset_dtype_degree_phrase_path = os.path.join(dataset_dtype_degree_path, phrase)
                target_phrase_path = os.path.join(target_dataset_path, phrase)
                
                # 获取要选择的图片数量
                num_to_select = select_num.get(phrase, 0)
                if num_to_select == 0:
                    continue
                    
                # 遍历红外光，可见光
                selected_indices = []
                for image_type in os.listdir(dataset_dtype_degree_phrase_path):
                    dataset_dtype_degree_phrase_image_type = os.path.join(dataset_dtype_degree_phrase_path, image_type)
                    target_phrase_image_type = os.path.join(target_phrase_path, image_type)
                    fileList = os.listdir(dataset_dtype_degree_phrase_image_type)
                    
                    # 检查是否有足够图片可供选择
                    if len(fileList) < num_to_select:
                        print(f'Warning: Not enough images in {dataset_dtype_degree_phrase_image_type}. '
                              f'Only {len(fileList)} available, need {num_to_select}.')
                        num_to_select = len(fileList)
                    
                    # 随机选择指定数量的图片
                    if len(selected_indices) == 0:
                        selected_indices = random.sample(range(len(fileList)), num_to_select)
                    
                    # 创建目标目录
                    os.makedirs(target_phrase_image_type, exist_ok=True)
                    
                    # 复制选中的图片到目标目录
                    copied_count = 0
                    for index in selected_indices:
                        file = fileList[index]
                        filename = file.split('.')[0]
                        degree_ = degree.split('_')[-1]
                        dtype_ = dtype.split('_')[-1]
                        new_filename = filename + '_' + degree_ + '_' + dtype_ + '.' + save_ext
                        src_path = os.path.join(dataset_dtype_degree_phrase_image_type, file)
                        dst_path = os.path.join(target_phrase_image_type, new_filename)
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                        except Exception as e:
                            print(f'Error copying {src_path} to {dst_path}: {e}')
                    
                    print(f'  Selected and copied {copied_count} images to {target_phrase_image_type}')
    
    print(f'\nDataset construction completed! New dataset saved to {target_dataset_path}')

def move_multi():
    dataset_path = './dataset/DDL-12/Multi'
    '''
    对于复合退化 考虑到单个退化训练大概1000张 训练大概180张 因此直接将复合退化的全部图像移动到新文件中
    '''
    for dtype in os.listdir(dataset_path):

        dataset_dtype = os.path.join(dataset_path, dtype)

        for phrase in os.listdir(dataset_dtype):
            dataset_dtype_phrase = os.path.join(dataset_dtype, phrase)
            target_phrase = os.path.join(target_dataset_path, phrase)

            for image_type in os.listdir(dataset_dtype_phrase):
                dataset_dtype_phrase_image_type = os.path.join(dataset_dtype_phrase, image_type)
                target_phrase_image_type = os.path.join(target_phrase, image_type)

                copied_count = 0
                for file in os.listdir(dataset_dtype_phrase_image_type):
                    filename = file.split('.')[0]
                    compose_dtype = dtype.split('_')[1] + '_' + dtype.split('_')[2]
                    new_filename = filename + '_' + compose_dtype + '.' + save_ext
                    
                    src_path = os.path.join(dataset_dtype_phrase_image_type, file)
                    dst_path = os.path.join(target_phrase_image_type, new_filename)

                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f'Error copying {src_path} to {dst_path}: {e}')
                
                print(f'  Selected and copied {copied_count} images to {target_phrase_image_type}')

if __name__ == '__main__':
    # move_single()
    # move_multi()