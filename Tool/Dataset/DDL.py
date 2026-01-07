import shutil
import os

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
                    
