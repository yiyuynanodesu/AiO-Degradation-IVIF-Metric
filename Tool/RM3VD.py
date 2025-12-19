"""
merge_RM3DV : 合并退化的指定视频号的红外光和可见光
rename_RM3DV : 更改文件名 格式：文件名_退化_视频号.扩展名
避免合并到同一个文件的时候冲突
"""

import os
import shutil

from_path = './RM3DV'
target_path = './target_dataset'

itype_dict = {
    'vi': 'Visible',
    'ir': 'Infrared'
}

dtype_video_dict = {
    'night_rain': ['a'],
    'fog': ['a'],
    'day_rain': ['c']
}

def rename_RM3DV():
    for dtype in os.listdir(from_path):
        dtype_path = os.path.join(from_path, dtype)
        for itype in os.listdir(dtype_path):
            itype_dtype = os.path.join(dtype_path, itype)
            for video in os.listdir(itype_dtype):
                if video not in dtype_video_dict[dtype]:
                    continue
                video_itype_dtype = os.path.join(itype_dtype, video)
                for file in os.listdir(video_itype_dtype):
                    filename = file.split('.')[0]
                    ext = file.split('.')[-1]
    
                    new_filename = filename + '_' + dtype + '_' + video + '.' + ext
                    shutil.move(os.path.join(video_itype_dtype, file), os.path.join(video_itype_dtype, new_filename)) # pay attention, use shutil.move()
 
def merge_RM3DV():
    for dtype in os.listdir(from_path):
        dtype_path = os.path.join(from_path, dtype)
        for itype in os.listdir(dtype_path):
            itype_dtype = os.path.join(dtype_path, itype)
            itype_target = os.path.join(target_path, itype_dict[itype])
            for video in os.listdir(itype_dtype):
                if video not in dtype_video_dict[dtype]:
                    continue
                video_itype_dtype = os.path.join(itype_dtype, video)
                for file in os.listdir(video_itype_dtype):
                    filename = file.split('.')[0]
                    ext = file.split('.')[-1]
    
                    new_filename = filename + '_' + dtype + '_' + video + '.' + ext
                    shutil.copy(os.path.join(video_itype_dtype, file), os.path.join(itype_target, new_filename))
    
    print('done')

if __name__ == '__main__':
    # merge_RM3DV()
    rename_RM3DV()
            