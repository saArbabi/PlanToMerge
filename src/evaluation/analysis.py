import matplotlib.pyplot as plt
import json
import pickle
import os
import numpy as np


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

indexs = {}
metric_labels = ['budget', 'epi sode', 'cumulative_reward', 'timesteps_to_merge', \
                  'max_decision_time', 'hard_brake_count', 'decisions_made']

for i in range(6):
    indexs[metric_labels[i]] = i
indexs
# %%
planner_names = ["mcts", "omniscient", "qmdp"]
planner_names = ["mcts", "omniscient"]
planner_names = ["mcts_rand", "mcts_know", "omniscient"]
planner_names = ["mcts_rand", "mcts_know", "omniscient", "qmdp"]
# planner_names = ["mcts"]
planner_names = ["mcts_step_1", "mcts_step_2"]
planner_names = ["mcts_step1", "mcts_step2"]
planner_names = ["qmdp_simple", "qmdp_clever", "qmdp_clever2"]
planner_names = ["qmdp_clever", "qmdp_clever2"]
planner_names = ["qmdp_simple", "qmdp_clever", "qmdp_clever2", "qmdp_clever3", "qmdp_clever4"]
planner_names = ['mcts_', "qmdp_10", "qmdp_30", 'qmdp']
planner_names = ['mcts', "qmdp"]

metric_dict = {}
decision_logs = {}
for planner_name in planner_names:
    metrics = []
    decision_logs[planner_name] = {}
    exp_dir = './src/evaluation/experiments/'+planner_name
    budgets, exp_names = get_budgets(exp_dir)
    mc_collection = get_mc_collection(exp_dir, exp_names)
    for i, budget in enumerate(budgets):
        decision_logs[planner_name][budget] = []
        budget_metric = []
        for episode, epis_metric in mc_collection[i].items():
            # if episode != 501:
            #     continue
            decision_logs[planner_name][budget] += epis_metric[-1]
            budget_metric.append([budget, episode]+epis_metric[0:-1])
        metrics.append(budget_metric)

    metric_dict[planner_name] = np.array(metrics)
    print(planner_name,' shape: ', metric_dict[planner_name].shape)

# metric_dict[planner_name].shape
# metric_dict[planner_name].shape
# dims: [budgets, episodes, logged_states]

# %%
subplot_xcount = 2
subplot_ycount = 2
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(10, 8))
axs[1, 0].set_xlabel('Iterations')
axs[1, 1].set_xlabel('Iterations')
for ax in axs.flatten():
    ax.set_xticks([10, 30])


def add_plot_to_fig(metrics, ax, kpi):
    # ax.plot(x_y[0], x_y[1][:, -1], \
    #                'o-', label=planner_name)
    if planner_name == 'omniscient':
        ax.plot(x_y[0], x_y[1].mean(axis=1), \
                       'o-', label=planner_name, color='red')
    else:
        ax.plot(    x_y[0], x_y[1].mean(axis=1), \
                       'o-', label=planner_name)
    ax.legend()
    ax.set_title(kpi)

for planner_name in planner_names:
    metrics = metric_dict[planner_name]

    kpi = 'cumulative_reward'
    x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
    ax = axs[0, 0]
    add_plot_to_fig(x_y, ax, kpi)
    kpi = 'timesteps_to_merge'
    x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
    ax = axs[0, 1]
    add_plot_to_fig(x_y, ax, kpi)
    kpi = 'max_decision_time'
    x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
    ax = axs[1, 0]
    add_plot_to_fig(x_y, ax, kpi)
    kpi = 'hard_brake_count'
    x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
    ax = axs[1, 1]
    add_plot_to_fig(x_y, ax, kpi)

# %%
"""
Performance comparison for each episode.
"""
fig, ax = plt.subplots(figsize=(5, 10))
episodes_considered = range(500, 500+metric_dict[planner_name].shape[1])
planner_count = len(planner_names)

y_pos = np.linspace(planner_count/2, (planner_count+2)*len(episodes_considered), len(episodes_considered))
y_pos
kpi = 'cumulative_reward'
# kpi = 'timesteps_to_merge'

for i, planner_name in enumerate(planner_names):
    metrics = metric_dict[planner_name]
    kpi_val = metrics[1, :, indexs[kpi]]
    ax.barh(y_pos + i, kpi_val, label=planner_name)

# metric_dict['mcts'][1, :, indexs[kpi]]
# metric_dict['omniscient'][1, :, indexs[kpi]]

labels = [str(epis) for epis in episodes_considered]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.spines['left'].set_position('zero')
ax.legend()
ax.grid()
# plt.plot([0, 0], \
#           [-1, 30], color = 'black')


# %%
s


x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
# %%
metric_dict['omniscient'][0, :, indexs['timesteps_to_merge']]
metric_dict['omniscient'][0, :, indexs['cumulative_reward']]
metric_dict['mcts'][0, :, indexs['cumulative_reward']]

metric_dict['mcts'][:, :, indexs['timesteps_to_merge']]
metric_dict['omniscient'][:, :, indexs['timesteps_to_merge']]
metric_dict['omniscient'][:, :, 3].mean(axis=1)
metric_dict['omniscient'][:, 0, 3]
metric_dict['omniscient'][:, 1, 3]
# %%

planner_name = 'omniscient'
exp_dir = './src/evaluation/experiments/'+planner_name
budgets, exp_names = get_budgets(exp_dir)
# %%
episode_id = 500
# %%
for planner_name in planner_names:
    timesteps_to_merge = [metrics[episode_id][0] for metrics in exp_logs[planner_name]['mc_collection']]
    budgets = exp_logs[planner_name]['budgets']
    plt.plot(budgets, timesteps_to_merge, 'o-', label=planner_name)
plt.xlabel('Budget')
plt.ylabel('Cummulative reward')
plt.legend()
plt.grid()

# %%
for planner_name in planner_names:
    timesteps_to_merge = [metrics[episode_id][1] for metrics in exp_logs[planner_name]['mc_collection']]
    budgets = exp_logs[planner_name]['budgets']
    plt.plot(budgets, timesteps_to_merge, 'o-', label=planner_name)
plt.xlabel('Budget')
plt.ylabel('TTM (steps)')
plt.legend()
plt.grid()

# %%
timesteps_to_merge = [metrics[episode_id][1] for metrics in mc_collection]
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
