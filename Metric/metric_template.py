from evaluate.eval_metric import eval_batch as normal_eval
from evaluate.degrad_eval_metric import eval_batch as degrad_eval
from evaluate.degrad_eval_metric_by_type import eval_batch as degrad_type_eval

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--model', default = '', help='model')
parser.add_argument('--dataset', default = '', help='dataset')
parser.add_argument('--ir_path', default = '', help='ir path')
parser.add_argument('--vi_path', default = '', help='vi path')
parser.add_argument('--result_path', default = '', help='result path')
parser.add_argument('--save_path', default = './', help='model')
args = parser.parse_args()

# Degrad IVIF
if args.dataset == 'DDL':
    # DDL
    degrad_eval(args.result_path, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_degrad.xlsx')
    # Degrad Type IVIF
    degrad_type_eval(args.result_path, args.save_path, degard_list=['HazeRain','HazeLow','Rain','Haze','Exposure','Light'], model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_detail.xlsx')
elif args.dataset == 'EMS':
    degrad_eval(args.result, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_degrad.xlsx')
    degrad_type_eval(args.result, args.save_path, degard_list=['Rain','Haze','Exposure','Light'], model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_detail.xlsx')
else:
    # Normal IVIF
    normal_eval(args.ir_path, args.vi_path, args.result_path, args.save_path, model_name=args.model, excel_filename=f'{args.model}_{args.dataset}_normal.xlsx')