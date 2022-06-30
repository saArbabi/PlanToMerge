"""
Includes:
1) Reward vs budget
2) TTM vs budget
3) Abortion vs budget
4) Agent's tendencies timid >>> aggressive (bar chart)
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
planner_names = ["mcts", "qmdp", "belief_search", "omniscient"]
planner_labels = ["MCTS", "QMDP", "Belief Search", "Omniscient"]

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
            decision_logs[planner_name][budget] += epis_metric[-1]
            budget_metric.append([budget, episode]+epis_metric[0:-1])
        metrics.append(budget_metric)

    metric_dict[planner_name] = np.array(metrics)
    print(planner_name,' shape: ', metric_dict[planner_name].shape)

indexs = {}
metric_labels = ['budget', 'episode', 'cumulative_reward', 'timesteps_to_merge', \
                  'max_decision_time', 'hard_brake_count', 'decisions_made']

for i in range(6):
    indexs[metric_labels[i]] = i
indexs
# %%
plt.figure(size=)
kpi = 'cumulative_reward'
for i, planner_name in enumerate(planner_names):
    metrics = metric_dict[planner_name]
    x_y = [metrics[:, 0, indexs['budget']], metrics[:, :, indexs[kpi]]]
    plt.plot(x_y[0], x_y[1].mean(axis=1), label=planner_labels[i])
plt.xlabel('Iterations')
plt.ylabel('Accumulated reward')
plt.grid()
plt.legend()

# %%
