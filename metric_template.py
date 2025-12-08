from Metric.eval_metric import eval_batch

ir_path = './dataset/MSRS/test/ir'
vis_path = './dataset/MSRS/test/vi'
output_path = './SPGFusion/OUTPUT/Time_test3/'
save_dir = './'

eval_batch(ir_path, vis_path, output_path, save_dir)