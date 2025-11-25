# All in one degradation infrared and visible image fusion

## Environment

```python
pip install scipy
pip install scikit-learn
pip install openpyxl
```

## Usage

use this template or see **metric_template.py**

```python
from Metric.eval_metric import eval_batch

ir_path = ''
vi_path = ''
output_path = ''
save_dir = ''

eval_batch(ir_path=ir_path, vis_path=vi_path, output_path=output_path, save_dir=save_dir)
```

or

```python
python metric_template.py
```

## Warning List
### SPGFusion
if you use SPGFusion, add this to eval_metric:
```python
w_vi, h_vi = image_vis.size  # (width, height)
w_ir, h_ir = image_ir.size
new_w = max(16, (w_vi // 16) * 16)
new_h = max(16, (h_vi // 16) * 16)
if (w_vi != new_w) or (h_vi != new_h):
    image_vis = image_vis.resize((new_w, new_h), resample=Image.BICUBIC)
if (w_ir != new_w) or (h_ir != new_h):
    image_ir = image_ir.resize((new_w, new_h), resample=Image.BICUBIC)
```

## Acknowledgement

Our code is based on the following 

[RollingPlain/IVIF_ZOO: Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption. A fire-new survey for infrared and visible image fusion.](https://github.com/RollingPlain/IVIF_ZOO)

