# All in one degradation infrared and visible image fusion

## Environment

```python
# first you need to prepare clip-iqa enviroment, see Metric
pip install scipy
pip install scikit-learn
pip install openpyxl
```

## Usage

use this template or see `Metric/metric_template.py`

```python
from evaluate.eval_metric import eval_batch as normal_eval
from evaluate.degrad_eval_metric import eval_batch as degrad_eval
from evaluate.degrad_eval_metric_by_type import eval_batch as degrad_type_eval

model = ''

# Normal IVIF
ir_path = ''
vis_path = ''
output_path = f'./{model}/'
save_dir = ''

normal_eval(ir_path, vis_path, output_path, save_dir, model_name=model, excel_filename=f'{model}_normal.xlsx')

# Degrad IVIF

# DDL Dataset
output_path = f'../{model}/'
save_dir = './'
degrad_eval(output_path, save_dir, model_name=model, excel_filename=f'{model}_degrad.xlsx')
# Degrad Type IVIF
degrad_type_eval(output_path, save_dir, degard_list=['HazeRain','HazeLow','Rain','Haze','Exposure','Light'], model_name=model, excel_filename=f'{model}_detail.xlsx')

# EMS Dataset
output_path = f'./{model}/'
save_dir = './'
degrad_eval(output_path, save_dir, model_name=model, excel_filename=f'{model}_degrad.xlsx')
degrad_type_eval(output_path, save_dir, degard_list=['Rain','Haze','Exposure','Light'], model_name=model, excel_filename=f'{model}_detail.xlsx')
```

or

```python
python metric_template.py
```

## Warning List
### SPGFusion
if you use SPGFusion, add this to eval_metric.py's line 36:
```python
w_vi, h_vi = vi_img.size  # (width, height)
w_ir, h_ir = ir_img.size
new_w = max(16, (w_vi // 16) * 16)
new_h = max(16, (h_vi // 16) * 16)
if (w_vi != new_w) or (h_vi != new_h):
    vi_img = vi_img.resize((new_w, new_h), resample=Image.BICUBIC)
if (w_ir != new_w) or (h_ir != new_h):
    ir_img = ir_img.resize((new_w, new_h), resample=Image.BICUBIC)
```

## Acknowledgement

Our code is based on the following:

[LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://github.com/bupt-ai-cz/LLVIP)

[Exploring CLIP for Assessing the Look and Feel of Images (AAAI 2023)](https://github.com/IceClear/CLIP-IQA/blob/v1-3.6/demo/clipiqa_single_image_demo.py)

[RollingPlain/IVIF_ZOO: Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption. A fire-new survey for infrared and visible image fusion.](https://github.com/RollingPlain/IVIF_ZOO)

