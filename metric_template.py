from Metric.eval_metric import eval_batch as normal_eval
from Metric.degrad_eval_metric import eval_batch as degrad_eval
from Metric.degrad_eval_metric_by_type import eval_batch as degrad_type_eval

model = ''

# Normal IVIF
ir_path = ''
vis_path = ''
output_path = f'./{model}/'
save_dir = ''

normal_eval(ir_path, vis_path, output_path, save_dir, model_name=model, excel_filename=f'{model}_normal.xlsx')

# Degrad IVIF

output_path = ''
save_dir = ''
degrad_eval(output_path, save_dir, model_name=model, excel_filename=f'{model}_degrad.xlsx')

# Degrad Type IVIF
output_path = ''
save_dir = ''
degrad_type_eval(output_path, save_dir, degard_list=['HazeRain','HazeLow','Rain','Haze','Exposure','Light'], model_name=model, excel_filename=f'{model}_detail.xlsx')