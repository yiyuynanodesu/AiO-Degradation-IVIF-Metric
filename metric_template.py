from Metric.eval_metric import eval_batch

# ir_path = './dataset/LLVIP/infrared/test'
# vi_path = './dataset/LLVIP/visible/test'
# output_path = './GIFNet/outputs/LLVIP_Result'
# save_dir = './'

# ir_path = './dataset/FMB/test/Infrared'
# vi_path = './dataset/FMB/test/Visible'
# output_path = './GIFNet/outputs/FMB_Result'
# save_dir = './'

# ir_path = './dataset/MSRS/test/ir'
# vi_path = './dataset/MSRS/test/vi'
# output_path = './GIFNet/outputs/MSRS_Result'
# save_dir = './'

# ir_path = './dataset/IR_LightDDL_Test_Split/Exposure/Infrared'
# vi_path = './dataset/IR_LightDDL_Test_Split/Exposure/Visible'
# output_path = './UP-Fusion/Exposure_Result'
# save_dir = './Exposure_Result'

ir_path = './dataset/IR_LightDDL_Test_Split/Haze/Infrared'
vi_path = './dataset/IR_LightDDL_Test_Split/Haze/Visible'
output_path = './UP-Fusion/Haze_Result'
save_dir = './'

# ir_path = './dataset/IR_LightDDL_Test_Split/HazeLowLight/Infrared'
# vi_path = './dataset/IR_LightDDL_Test_Split/HazeLowLight/Visible'
# output_path = './TITA/results_/LightDDL'
# save_dir = './'

# ir_path = './dataset/IR_LightDDL_Test_Split/Rain/Infrared'
# vi_path = './dataset/IR_LightDDL_Test_Split/Rain/Visible'
# output_path = './TITA/results_/LightDDL'
# save_dir = './'

# ir_path = './dataset/IR_LightDDL_Test_Split/LowLight/Infrared'
# vi_path = './dataset/IR_LightDDL_Test_Split/LowLight/Visible'
# output_path = './TITA/results_/LightDDL'
# save_dir = './'


# ir_path = './dataset/IR_LightDDL_Test_Split/HazeRain/Infrared'
# vi_path = './dataset/IR_LightDDL_Test_Split/HazeRain/Visible'
# output_path = './TITA/results_/LightDDL'
# save_dir = './'


eval_batch(ir_path=ir_path, vis_path=vi_path, output_path=output_path, save_dir=save_dir)