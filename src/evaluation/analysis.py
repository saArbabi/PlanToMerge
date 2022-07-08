import matplotlib.pyplot as plt
import json
import pickle
import os
import numpy as np


# %%
def get_mc_collection(exp_dir, exp_names):
    mc_collection = []
    for exp_name in exp_names:
        with open(exp_dir+'/'+exp_name, 'rb') as handle:
            mc_collection.append(pickle.load(handle))
    return mc_collection

def get_budgets(exp_dir):
    exp_names = os.listdir(exp_dir)
    budgets = []
    for exp_name in exp_names:
        budget = [int(s) for s in exp_name[:-7].split('-') if s.isdigit()]
        budgets += budget
    budgets, exp_names = zip(*sorted(zip(budgets, exp_names)))
    return budgets, exp_names


indexs = {}
metric_labels = ['got_bad_state', 'cumulative_reward', 'timesteps_to_merge', \
                  'max_decision_time', 'hard_brake_count', 'decisions_made', 'agent_aggressiveness']

for i in range(7):
    indexs[metric_labels[i]] = i
indexs
# %%

planner_names = ["mcts", "mcts_mean", "qmdp", "belief_search", "omniscient"]
# planner_names = ["qmdp"]
# run_name = 'run317'
run_name = 'run_23'
# run_name = 'run_22'

decision_logs = {}
aggressiveness_logs = {}
metric_logs = {}
for planner_name in planner_names:
    decision_logs[planner_name] = {}
    aggressiveness_logs[planner_name] = {}
    metric_logs[planner_name] = {}
    exp_dir = './src/evaluation/experiments/'+run_name+'/'+planner_name
    budgets, exp_names = get_budgets(exp_dir)
    mc_collection = get_mc_collection(exp_dir, exp_names)
    for i, budget in enumerate(budgets):
        decision_logs[planner_name][budget] = {}
        aggressiveness_logs[planner_name][budget] = {}
        metric_logs[planner_name][budget] = {}
        for episode, epis_metric in mc_collection[i].items():
            # if episode > 509:
            #     continue
            decision_logs[planner_name][budget][episode] = epis_metric[-2]
            aggressiveness_logs[planner_name][budget][episode] = epis_metric[-1]
            metric_logs[planner_name][budget][episode] = epis_metric[0:-2]

# metric_dict[planner_name].shape
# metric_dict[planner_name].shape
# dims: [budgets, episodes, logged_states]
metric_logs['omniscient'][50]
# %%

# %%
subplot_xcount = 2
subplot_ycount = 2
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(10, 8))
axs[1, 0].set_xlabel('Iterations')
axs[1, 1].set_xlabel('Iterations')
for ax in axs.flatten():
    ax.set_xticks([50, 100, 150, 200])
    ax.grid()

def add_plot_to_fig(plot_data, ax, kpi):
    plot_data = budgets, metrics_arrs
    metrics_mean = [metrics_arr[:, indexs[kpi]].mean(axis=0) for metrics_arr in metrics_arrs]
    if planner_name == 'omniscient':
        ax.plot(budgets, metrics_mean, \
                       'o-', label=planner_name, color='red')
    else:
        ax.plot(budgets, metrics_mean, \
                       'o-', label=planner_name)
    ax.legend()
    ax.set_title(kpi)

for planner_name in planner_names:
    metrics = metric_logs[planner_name]
    budgets = list(metrics.keys())
    # budgets = list(metrics.keys())[1:]
    metrics_arrs = []
    for budget in budgets:
        episodes = metrics[budget].keys()
        metrics_arrs.append(np.array(list(metrics[budget].values())))

    kpi = 'cumulative_reward'
    ax = axs[0, 0]
    add_plot_to_fig([budgets, metrics_arrs], ax, kpi)

    kpi = 'timesteps_to_merge'
    ax = axs[0, 1]
    metrics_arrs = [metrics_arr[metrics_arr[:, indexs['got_bad_state']] != 1] for metrics_arr in metrics_arrs]
    add_plot_to_fig([budgets, metrics_arrs], ax, kpi)

    kpi = 'max_decision_time'
    ax = axs[1, 0]
    add_plot_to_fig([budgets, metrics_arrs], ax, kpi)

    kpi = 'hard_brake_count'
    ax = axs[1, 1]
    add_plot_to_fig([budgets, metrics_arrs], ax, kpi)
# %%
"""
Agent aggressiveness distriubiton
"""
budget = 50
bins = 30
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728']

for color, planner_name in zip(colors, planner_names):
    agg = list(aggressiveness_logs[planner_name][budget].values())
    agg = [x for xs in agg for x in xs]
    _ = plt.hist(agg, bins=bins, label=planner_name, edgecolor=color, fill=False)

plt.legend()

# _ = plt.hist(true_collection70.flatten(), bins=50, alpha=0.5, label='Human', 'fill')

3 * 0.9**10
# %%
"""
Count number of agent decisions
"""
bins = 5
for color, planner_name in zip(colors, planner_names):
    decisions = list(decision_logs[planner_name][budget].values())
    decisions = [x for xs in decisions for x in xs]
    _ = plt.hist(decisions, bins=bins, label=planner_name, edgecolor=color, fill=False)
    plt.title(planner_name)

# %%
"""
Assess agent decisions
"""
planner_name = 'mcts'
budget = 50
# budget = 1
OPTIONS = {1 : ['LANEKEEP', 'UP'],
           2 : ['LANEKEEP', 'IDLE'],
           3 : ['LANEKEEP', 'DOWN'],
           4 : ['MERGE', 'IDLE'],
           5 : ['ABORT', 'IDLE']}


labels = [action[0]+'_'+action[1] for action in OPTIONS.values()]


for epis in range(515, 516):
    plt.figure()
    plt.grid()
    plt.title('episode: '+ str(epis))
    for planner_name in planner_names:
        plt.plot(decision_logs[planner_name][budget][epis], '--x', label=planner_name)
    plt.legend()
    plt.yticks(range(1, 6), labels)

    # plt.figure()
    # plt.grid()
    # plt.title('episode: '+ str(epis))
    # for planner_name in planner_names:
    #     plt.plot(aggressiveness_logs[planner_name][budget][epis], '--x', label=planner_name)
    # plt.legend()

# %%
3 * 0.9**0
a = [1, 2, 3, 4]
a = [i for i in a if i != 2]
a
"""
Performance comparison for each episode.
"""
fig, ax = plt.subplots(figsize=(10, 50))
# fig, ax = plt.subplots()
planner_count = len(planner_names)

budget = 50
episodes_considered = metric_logs[planner_name][budget].keys()

kpi = 'cumulative_reward'
# kpi = 'timesteps_to_merge'
# kpi = 'max_decision_time'

y_pos = np.linspace(planner_count/2, (planner_count+5)*len(episodes_considered), len(episodes_considered))
# planner_names = ['qmdp']
planner_names
for i, planner_name in enumerate(planner_names):
    metrics_arr = np.array(list(metric_logs[planner_name][budget].values()))
    kpi_val = metrics_arr[:, indexs[kpi]]
    ax.barh(y_pos + i, kpi_val, label=planner_name)


labels = [str(epis) for epis in episodes_considered]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.spines['left'].set_position('zero')
ax.legend()
ax.grid()


# %%
s



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
