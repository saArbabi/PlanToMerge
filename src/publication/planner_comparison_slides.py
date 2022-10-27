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


def get_metric(metrics, kpi, planner_name):
    if kpi == 'timesteps_to_merge':
        clean_metrics = {}
        for budget in metrics.keys():
            _clean_metrics = {}
            for epis in metrics[budget].keys():
                if metrics[budget][epis][indexs['got_bad_state']] != 1:
                    _clean_metrics[epis] = metrics[budget][epis]
            clean_metrics[budget] = _clean_metrics
        metrics = clean_metrics


    if planner_name == 'rule_based':
        metric = [np.array(list(metrics[1].values()))[:, indexs[kpi]] for budget in budgets]
    else:
        metric = [np.array(list(metrics[budget].values()))[:, indexs[kpi]] for budget in budgets]

    return metric

def get_giveway_rates():
    """
    Return average abort rate across budgets
    """
    giveway_rates = []
    for budget in budgets:
        budget_giveway_rates = []
        for epis_decisions in list(decision_logs[planner_name][budget].values()):
            if max(epis_decisions) == 5:
                budget_giveway_rates.append(1)
            else:
                budget_giveway_rates.append(0)
        giveway_rates.append(np.mean(budget_giveway_rates))
    return giveway_rates

planner_name = 'mcts'
metrics = metric_logs[planner_name]
metric = get_metric(metrics, kpi, planner_name)

# %%


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
run_name = 'run_39'
planner_names = ["rule_based", "omniscient", "mcts", "belief_search"]
planner_labels = ["Rule-based", "Omniscient", "MCTS", "LVT"]


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
order = [2, 3, 1, 0]

colors = ['black', 'red','blue', 'darkgreen']
line_styles = ['--', '-o', '-s', '-D']

kpi = 'timesteps_to_merge'
budgets = list(metric_logs['omniscient'].keys())
plt.figure()
plt.xscale('log', basex=2)
plt.xlim(0, 520)
plt.ylim(14.5, 18.5)
plt.xlabel('Iterations')
plt.ylabel('Time to merge (s)')
plt.grid(alpha=0.5)
plt.xticks(budgets)

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metrics = metric_logs[planner_name]
    metric = get_metric(metrics, kpi, planner_name)
    metric = [_metric/10 for _metric in metric]
    metric_avgs = [_metric.mean() for _metric in metric]
    metric_std = [_metric.std()/10 for _metric in metric]
    if planner_name == 'rule_based':
        plt.plot([-1, 600], [metric_avgs[0], metric_avgs[0]], line_style, color=color, label=planner_label)
    else:
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

    # plt.errorbar(budgets, metric_avgs, metric_std)

    plt.savefig('ttm_'+planner_name+'.pdf', dpi=500, bbox_inches='tight')






# %%
"""
Reward to merge vs budget
"""
kpi = 'cumulative_reward'
plt.xscale('log', basex=2)

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
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), edgecolor='black', ncol=3)
plt.grid(alpha=0.5)
plt.savefig("planner_rewards.pdf", dpi=500, bbox_inches='tight')

# %%
"""
giveway count (%) vs budget
"""
plt.xscale('log', basex=2)

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    if planner_name != 'rule_based':
        metric_avgs = get_giveway_rates()
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iterations')
plt.ylabel('Fraction of giveway decisions')
plt.xticks(budgets)
plt.grid(alpha=0.5)
plt.savefig("giveway_rate.pdf", dpi=500, bbox_inches='tight')

# %%
"""
Decision time
"""
plt.figure()
plt.xscale('log', basex=2)
kpi = 'max_decision_time'
for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    if planner_name in ['mcts', 'belief_search']:
        metrics = metric_logs[planner_name]
        metric = get_metric(metrics, kpi, planner_name)
        metric_avgs = [_metric.mean() for _metric in metric]
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iterations')
plt.ylabel('Average Decision Time (s)')
plt.xticks(budgets)
plt.grid(alpha=0.5)
plt.savefig("avg_decision_time.pdf", dpi=500, bbox_inches='tight')

# %%
decision_logs
