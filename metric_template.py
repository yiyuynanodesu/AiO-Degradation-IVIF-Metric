from Metric.eval_metric import eval_batch as normal_eval
from Metric.degrad_eval_metric import eval_bath as degrad_eval

# Normal IVIF
ir_path = ''
vis_path = ''
output_path = ''
save_dir = ''

normal_eval(ir_path, vis_path, output_path, save_dir, model_name='Model', excel_filename='metric.xlsx')

# Degrad IVIF

output_path = ''
save_dir = ''
degrad_eval(output_path, save_dir, model_name='Model', excel_filename='metric.xlsx')