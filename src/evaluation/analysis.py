import matplotlib.pyplot as plt
import json
import pickle
import os


# %%

# %%

eval_config = read_eval_config()

exp_names
# %%

eval_config_dir = './src/evaluation/experiments/eval_config.json'
def get_mc_collection(exp_dir, exp_names):
    mc_collection = []
    for exp_name in exp_names:
        with open(exp_dir+'/'+exp_name, 'rb') as handle:
            mc_collection.append(pickle.load(handle))
    return mc_collection

def read_eval_config():
    with open(eval_config_dir, 'rb') as handle:
        eval_config = json.load(handle)
    return eval_config

def get_budgets(exp_dir):
    exp_names = os.listdir(exp_dir)
    budgets = []
    for exp_name in exp_names:
        budget = [int(s) for s in exp_name[:-7].split('-') if s.isdigit()]
        budgets += budget
    budgets, exp_names = zip(*sorted(zip(budgets, exp_names)))
    return budgets, exp_names

planner_names = ['mcts', 'omniscient']
exp_logs = {}
for planner_name in planner_names:
    exp_logs[planner_name] = {}

for planner_name in planner_names:
    # planner_name = 'mcts'
    exp_dir = './src/evaluation/experiments/'+planner_name
    budgets, exp_names = get_budgets(exp_dir)
    mc_collection = get_mc_collection(exp_dir, exp_names)
    exp_logs[planner_name]['exp_names'] = exp_names
    exp_logs[planner_name]['budgets'] = budgets
    exp_logs[planner_name]['mc_collection'] = mc_collection
exp_logs[planner_name]['mc_collection']

# %%\
planner_name = 'omniscient'
exp_dir = './src/evaluation/experiments/'+planner_name
budgets, exp_names = get_budgets(exp_dir)
# %%
for planner_name in planner_names:
    timesteps_to_merge = [metrics[3][0] for metrics in exp_logs[planner_name]['mc_collection']]
    budgets = exp_logs[planner_name]['budgets']
    plt.plot(budgets, timesteps_to_merge, 'o-', label=planner_name)
plt.xlabel('Budget')
plt.ylabel('Cummulative reward')
plt.legend()
plt.grid()

# %%
for planner_name in planner_names:
    timesteps_to_merge = [metrics[3][1] for metrics in exp_logs[planner_name]['mc_collection']]
    budgets = exp_logs[planner_name]['budgets']
    plt.plot(budgets, timesteps_to_merge, 'o-', label=planner_name)
plt.xlabel('Budget')
plt.ylabel('TTM (steps)')
plt.legend()
plt.grid()

# %%
timesteps_to_merge = [metrics[3][1] for metrics in mc_collection]
plt.plot(budgets, timesteps_to_merge, 'o-', label='With yield car')
# plt.plot(budgets, [170]*7, 'o-', label='With yield car')
plt.xlabel('Budget')
plt.ylabel('TTM (steps)')
plt.legend()

# %%

'f{}'.

name = 'salar'
f'{name} is my name'
planner_name
planner_name = 'mcts'
budget = 20
f'{planner_name}-budget-{budget}'

0 ** 10
0.5/0.75
sum([4, 4])
