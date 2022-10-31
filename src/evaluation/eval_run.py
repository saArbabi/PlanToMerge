import sys
import os
import json
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVAL


def read_eval_config(config_path):
    with open(config_path, 'rb') as handle:
        eval_config = json.load(handle)
    return eval_config

def get_run_names():
    file_names = os.listdir('./src/evaluation/experiments')
    names = []
    for name in file_names:
        if name[:4] == 'run_':
            names.append(name)
    return names

def main():
    run_name = 'run_40'
    # run_name = 'run_test'
    names = get_run_names()
    if run_name in names:
        config_path = './src/evaluation/experiments/'+run_name+'/eval_config.json'
    else:
        config_path = './src/evaluation/experiments/eval_config.json'

    eval_config = read_eval_config(config_path)
    eval_obj = MCEVAL(eval_config)
    eval_obj.exp_dir = './src/evaluation/experiments/'+run_name
    eval_obj.eval_config_dir = eval_obj.exp_dir+'/eval_config.json'
    if not os.path.exists(eval_obj.exp_dir):
        os.mkdir(eval_obj.exp_dir)

    eval_obj.run()

if __name__=='__main__':
    main()
