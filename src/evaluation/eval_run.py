import sys
import os
import json
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVAL

eval_config_dir = './src/evaluation/experiments/eval_config.json'

def read_eval_config():
    with open(eval_config_dir, 'rb') as handle:
        eval_config = json.load(handle)
    return eval_config

def main():
    eval_config = read_eval_config()
    eval_obj = MCEVAL(eval_config)
    # eval_obj.mc_run_name = mc_run_name
    eval_obj.eval_config_dir = eval_config_dir
    eval_obj.run()

if __name__=='__main__':
    main()
