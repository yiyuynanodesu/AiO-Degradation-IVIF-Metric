import os
import shutil

from_path = './RM3DV'
target_path = './target_dataset'

itype_dict = {
    'vi': 'Visible',
    'ir': 'Infrared'
}

dtype_video_dict = {
    'night_rain': ['d'],
    'fog': ['o'],
    'day_rain': ['k']
}

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
            