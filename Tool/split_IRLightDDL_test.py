import os
import shutil

"""

对于IR_LightDDL的test数据集 把每个退化单独拿出来
最后变成

雾
雨
低光
高曝
雾+低光
雾+雨

"""

dataset_path = './dataset/IR_Light_DDL-12/test'
target_path = './target_dataset'

os.makedirs(target_path, exist_ok=True)

    
for image_type in os.listdir(dataset_path):
    dataset_image_type = os.path.join(dataset_path, image_type)
    target_image_type = os.path.join(target_path, image_type)
    os.makedirs(target_image_type, exist_ok=True)

    haze_path = os.path.join(target_image_type, "Haze")
    os.makedirs(haze_path, exist_ok=True)
    rain_path = os.path.join(target_image_type, "Rain")
    os.makedirs(rain_path, exist_ok=True)
    light_path = os.path.join(target_image_type, "LowLight")
    os.makedirs(light_path, exist_ok=True)
    exposure_path = os.path.join(target_image_type, "Exposure")
    os.makedirs(exposure_path, exist_ok=True)
    haze_light_path = os.path.join(target_image_type, "HazeLowLight")
    os.makedirs(haze_light_path, exist_ok=True)
    haze_rain_path = os.path.join(target_image_type, "HazeRain")
    os.makedirs(haze_rain_path, exist_ok=True)

    for file in os.listdir(dataset_image_type):
        if "Rain" in file and "Haze" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(haze_rain_path, file))
            continue
        if "Low" in file and "Haze" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(haze_light_path, file))
            continue
        if "Rain" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(rain_path, file))
            continue
        if "Haze" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(haze_path, file))
            continue
        if "exposure" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(exposure_path, file))
            continue
        if "light" in file:
            shutil.copy(os.path.join(dataset_image_type, file), os.path.join(light_path, file))
            continue
        print(f'!!!!!!! {file} not move!!!')
    
        

