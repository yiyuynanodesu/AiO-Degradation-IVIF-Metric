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

## Acknowledgement

Our code is based on the following 

[RollingPlain/IVIF_ZOO: Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption. A fire-new survey for infrared and visible image fusion.](https://github.com/RollingPlain/IVIF_ZOO)

