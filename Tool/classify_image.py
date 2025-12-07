import shutil
import os

"""
对 Light DDL-12中的图像进行分类 主要分为光照退化 及 天气退化 对于雾+低光单独拿出来
天气（雨，雾，雨+雾，雾+低光 使用OneRestore恢复)
光照（低光，高曝 使用InstructIR恢复）
参考ControlFusion
"""

dataset_path = './dataset/Light_DDL-12'
target_path = './target_dataset'

os.makedirs(target_path, exist_ok=True)

for phrase in os.listdir(dataset_path):
    dataset_phrase = os.path.join(dataset_path, phrase)
    target_phrase = os.path.join(target_path, phrase)
    os.makedirs(target_phrase, exist_ok=True)
    
    for image_type in os.listdir(dataset_phrase):
        dataset_phrase_image_type = os.path.join(dataset_phrase, image_type)
        target_phrase_image_type = os.path.join(target_phrase, image_type)
        os.makedirs(target_phrase_image_type, exist_ok=True)

        weather_target_phrase_image_type = os.path.join(target_phrase_image_type, "weather")
        os.makedirs(weather_target_phrase_image_type, exist_ok=True)
        light_target_phrase_image_type = os.path.join(target_phrase_image_type, "light")
        os.makedirs(light_target_phrase_image_type, exist_ok=True)

        for file in os.listdir(dataset_phrase_image_type):
            if "Rain" in file or "Haze" in file:
                shutil.copy(os.path.join(dataset_phrase_image_type, file), os.path.join(weather_target_phrase_image_type, file))
                continue
            if "light" in file or "exposure" in file:
                shutil.copy(os.path.join(dataset_phrase_image_type, file), os.path.join(light_target_phrase_image_type, file))
                continue
            print(f'!!!!!!! {file} not move!!!')
        
        