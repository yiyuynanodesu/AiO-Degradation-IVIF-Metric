import shutil
import os
import random

def move_data():
    from_path = './DDL-12/'
    to_path = './Dataset/'
    
    for degrad in os.listdir(from_path):
        degrad_path = os.path.join(from_path, degrad)
        for level in os.listdir(degrad_path):
            level_path = os.path.join(degrad_path, level)
            for phrase in os.listdir(level_path):
                phrase_path = os.path.join(level_path, phrase)
                to_phrase_path = os.path.join(to_path, phrase)
                for cls in os.listdir(phrase_path):
                    if cls == 'text.txt':
                        continue
                    cls_path = os.path.join(phrase_path, cls)
                    to_cls_path = os.path.join(to_phrase_path, cls)
                    for filename in os.listdir(cls_path):
                        basefilename = filename.split('.')[0]
                        ext = filename.split('.')[1]
                        now_degrad = degrad.split('_')[-1]
                        now_level = level.split('_')[-1]
                        save_filename = basefilename + '_' + now_degrad + '_' + now_level + '.' + ext
                        shutil.copy(os.path.join(cls_path, filename), os.path.join(to_cls_path, save_filename))
    
    print('done')
    
def move_single():
    save_ext = 'png'
    dataset_path = './DDL-12'
    target_dataset_path = './DDL4'
    '''
    对于单个退化 从每个程度的训练集中选125张 测试集选30张 这样单个退化有500张训练数据 120张测试数据
    '''
    
    select_num = {
        'train': 125,
        'test': 30
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
                    if image_type == 'text.txt':
                        continue
                    
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
    对于复合退化 从单个复合退化训练集中随机选择400张 测试集中随机选择80张
    '''

    select_num = {
        'train': 400,  # 训练集选择400张
        'test': 80     # 测试集选择80张
    }
    
    for dtype in os.listdir(dataset_path):

        dataset_dtype = os.path.join(dataset_path, dtype)

        for phrase in os.listdir(dataset_dtype):
            dataset_dtype_phrase = os.path.join(dataset_dtype, phrase)
            target_phrase = os.path.join(target_dataset_path, phrase)

            num_to_select = select_num.get(phrase, 0)
            if num_to_select == 0:
                continue

            selected_indices = []
            for image_type in os.listdir(dataset_dtype_phrase):
                dataset_dtype_phrase_image_type = os.path.join(dataset_dtype_phrase, image_type)
                target_phrase_image_type = os.path.join(target_phrase, image_type)
                fileList = os.listdir(dataset_dtype_phrase_image_type)

                # 检查是否有足够图片可供选择
                if len(fileList) < num_to_select:
                    print(f'Warning: Not enough images in {dataset_dtype_degree_phrase_image_type}. '
                          f'Only {len(fileList)} available, need {num_to_select}.')
                    num_to_select = len(fileList)
                
                # 随机选择指定数量的图片
                if len(selected_indices) == 0:
                    selected_indices = random.sample(range(len(fileList)), num_to_select)

                copied_count = 0
                for index in selected_indices:
                    file = fileList[index]
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
    # move_data()
    move_single()