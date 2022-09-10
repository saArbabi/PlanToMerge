import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


os.getcwd()
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

def get_metric(metrics, kpi, planner_name):
    if planner_name == 'rule_based':
        metric = [np.array(list(metrics[1].values()))[:, indexs[kpi]] for budget in budgets]
    else:
        metric = [np.array(list(metrics[budget].values()))[:, indexs[kpi]] for budget in budgets]
    return metric

def get_abort_rates():
    """
    Return average abort rate across budgets
    """
    abort_rates = []
    for budget in budgets:
        budget_abort_rates = []
        for epis_decisions in list(decision_logs[planner_name][budget].values()):
            if max(epis_decisions) == 6:
                budget_abort_rates.append(1)
            else:
                budget_abort_rates.append(0)
        abort_rates.append(np.mean(budget_abort_rates))
    return abort_rates

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
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


# %%
""" Load sim collections
"""
run_name = 'run_23'
planner_names = ["mcts", "mcts_mean", "qmdp", "belief_search", "omniscient", "rule_based"]

planner_labels = ["MCTS", "Assume normal", "QMDP", "Belief Search", "Omniscient", "Rule-based policy"]


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
colors = ['blue', 'blue', 'darkgreen', 'darkgreen', 'red', 'black']
line_styles = ['-o', '--s', '--s', '-o', '-s', '-s']

kpi = 'timesteps_to_merge'


fig, ax = plt.subplots(figsize=(8, 5))
budgets = list(metric_logs['omniscient'].keys())

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metrics = metric_logs[planner_name]
    metric = get_metric(metrics, kpi, planner_name)
    metric = [_metric/10 for _metric in metric]
    metric_avgs = [_metric.mean() for _metric in metric]
    metric_std = [_metric.std()/2 for _metric in metric]
    plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iterations')
plt.ylabel('Time to merge (s)')
plt.xticks(budgets)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), edgecolor='black', ncol=3)
plt.grid(alpha=0.5)
plt.savefig("ttm.pdf", dpi=500, bbox_inches='tight')






# %%
"""
Reward to merge vs budget
"""
kpi = 'cumulative_reward'
fig, ax = plt.subplots(figsize=(8, 5))

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    if planner_name != 'rule_based':
        metrics = metric_logs[planner_name]
        metric = get_metric(metrics, kpi, planner_name)
        metric = [_metric for _metric in metric]
        metric_avgs = [_metric.mean() for _metric in metric]
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iterations')
plt.ylabel('Episode reward')
plt.xticks(budgets)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), edgecolor='black', ncol=3)
plt.grid(alpha=0.5)
plt.savefig("planner_rewards.pdf", dpi=500, bbox_inches='tight')

# %%
"""
Abort count (%) vs budget
"""
fig, ax = plt.subplots(figsize=(8, 5))

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    if planner_name != 'rule_based':
        metric_avgs = get_abort_rates()
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iterations')
plt.ylabel('Fraction of abort decisions')
plt.xticks(budgets)
plt.grid(alpha=0.5)
plt.savefig("abort_rate.pdf", dpi=500, bbox_inches='tight')

# %%
"""
Decision count vs planner (a bar chart)
"""


# %%
decision_logs
