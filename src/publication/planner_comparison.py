import os
import pickle
import matplotlib.pyplot as plt
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

""" plot setup
"""
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 13,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


# %%
""" Load sim collections
"""
run_name = 'run_23'
planner_names = ["mcts", "mcts_mean", "qmdp", "belief_search", "omniscient"]
planner_labels = ["MCTS", "Assume normal", "QMDP", "Belief Search", "Omniscient"]
colors = ['blue', 'blue', 'orange', 'green', 'red']
line_styles = ['-o', '--o', '-o', '-o', '-s']

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

# %%
"""
Time to merge vs budget
"""
colors = ['blue', 'blue', 'orange', 'green', 'red']
line_styles = ['-o', '--o', '-o', '-o', '-s']

kpi = 'timesteps_to_merge'
def get_metric(metrics, kpi):
    metric = [np.array(list(metrics[budget].values()))[:, indexs[kpi]] for budget in budgets]
    return metric

fig, ax = plt.subplots(figsize=(7, 5))
for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    budgets = list(metric_logs[planner_name].keys())
    metrics = metric_logs[planner_name]
    metric = get_metric(metrics, kpi)
    metric = [_metric/10 for _metric in metric]
    metric_avgs = [_metric.mean() for _metric in metric]
    metric_std = [_metric.std()/2 for _metric in metric]
    plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)
    # plt.errorbar(budgets, metric_avgs, metric_std)
"""
add some error bar

"""
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), edgecolor='black', ncol=3)




# %%

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metrics = metric_logs[planner_name]
    metrics_arrs = []
    for budget in budgets:
        episodes = metrics[budget].keys()
        metrics_arrs.append(np.array(list(metrics[budget].values())))

    kpi = 'timesteps_to_merge'
    ax = axs[0, 0]
    add_plot_to_fig([budgets, metrics_arrs], ax, kpi)

# %%
"""
Reward to merge vs budget
"""

# %%
"""
Abort count vs budget
"""

# %%
"""
Decision count vs planner (a bar chart)
"""

# %%
