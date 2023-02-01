import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib tk

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
                if metrics[budget][epis][indexs['got_bad_state']] != 1: # note these episodes are terminated
                    _clean_metrics[epis] = metrics[budget][epis]
            clean_metrics[budget] = _clean_metrics
        metrics = clean_metrics

    if planner_name == 'rule_based':
        metric = [np.array(list(metrics[1].values()))[:, indexs[kpi]] for budget in budgets]
    else:
        metric = [np.array(list(metrics[budget].values()))[:, indexs[kpi]] for budget in budgets]
    return metric

def get_giveway_rates(planner_name):
    """
    Return average abort rate across budgets
    """
    if planner_name != 'rule_based':
        giveway_rates = []
        for budget in budgets:
            budget_giveway_rates = []
            for epis_decisions in list(decision_logs[planner_name][budget].values()):
                budget_giveway_rates.append(epis_decisions.count(5)/len(epis_decisions))
                # if max(epis_decisions) == 5:
                # else:
                #     budget_giveway_rates.append(0)
            giveway_rates.append(np.mean(budget_giveway_rates))
    else:
        giveway_rates = []
        budget = 1
        budget_giveway_rates = []
        for epis_decisions in list(decision_logs[planner_name][budget].values()):
            if max(epis_decisions) == 5:
                budget_giveway_rates.append(1)
            else:
                budget_giveway_rates.append(0)
        giveway_rates = [np.mean(budget_giveway_rates)]*6
    return giveway_rates

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
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


# %%
""" Load sim collections
"""
# run_name = 'run_23'
# run_name = 'run_39'
run_name = 'run_69'
planner_names = ["mcts", "mcts_mean", "qmdp", "belief_search", "omniscient", "rule_based"]
planner_labels = ["MCTS", "MCTS-normal", "QMDP", "LVT", "Omniscient", "Rule-based"]
# planner_names = ["omniscient", "belief_search"]
# planner_labels = ["Omniscient", "LVT"]


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
line_styles = ['-o', '--d', '--d', '-o', '--s', '--']

kpi = 'timesteps_to_merge'



budgets = list(metric_logs['omniscient'].keys())
plt.xscale('log', basex=2)

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metrics = metric_logs[planner_name]
    metric = get_metric(metrics, kpi, planner_name)
    metric = [_metric/10 for _metric in metric]
    metric_avgs = [_metric.mean() for _metric in metric]
    if planner_name == 'rule_based':
        plt.plot([0, 2000], [metric_avgs[0], metric_avgs[0]], line_style, color=color, label=planner_label)
    elif planner_name in ['belief_search', 'mcts']:
        metric_std = [_metric.std()/np.sqrt(100) for _metric in metric]
        plt.errorbar(budgets, metric_avgs, metric_std, color=color, label=planner_label, capsize=5)
        plt.plot(budgets, metric_avgs, line_style, color=color)
    else:
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

    plt.xlim(0, 520)

plt.xlabel('Iteration n')
plt.ylabel('Time to merge (s)')
plt.xticks(budgets)
plt.xlim(0, 1500)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), edgecolor='black', ncol=3)
# plt.legend()
plt.grid(alpha=0.5)
plt.savefig("planner_ttm.pdf", dpi=500, bbox_inches='tight')



# %%
"""
got_bad_state vs budget
"""
kpi = 'got_bad_state'
plt.xscale('log', basex=2)

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metrics = metric_logs[planner_name]
    metric = get_metric(metrics, kpi, planner_name)
    metric = [_metric for _metric in metric]
    metric_avgs = [_metric.mean()*100 for _metric in metric]
    if planner_name == 'rule_based':
        plt.plot([0, 2000], [metric_avgs[0], metric_avgs[0]], line_style, color=color, label=planner_label)
    else:
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)
        print(planner_name, ' ', budgets[-1],' ', metric_avgs[-1])
        # plt.errorbar(budgets, metric_avgs, metric_std)

plt.xlabel('Iteration n')
plt.ylabel('Unsafe state rate (%)')
plt.xticks(budgets)
plt.xlim(0, 1500)
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("planner_got_bad_state.pdf", dpi=500, bbox_inches='tight')

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
        # plt.errorbar(budgets, metric_avgs, metric_std)

plt.xlabel('Iteration n')
plt.ylabel('Episode reward')
plt.xticks(budgets)
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("planner_rewards.pdf", dpi=500, bbox_inches='tight')

# %%
"""
Abort count (%) vs budget
"""
plt.xscale('log', basex=2)

for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    metric_avgs = get_giveway_rates(planner_name)

    if planner_name != 'rule_based':
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

    # if planner_name == 'rule_based':
        # plt.plot([0, 1024], [metric_avgs[0], metric_avgs[0]], line_style, color=color, label=planner_label)

    # elif planner_name in ['belief_search', 'mcts']:
    #     metric_std = [_metric.std()/np.sqrt(100) for _metric in metric]
    #     plt.errorbar(budgets, metric_avgs, metric_std, color=color, label=planner_label, capsize=5)
    #     plt.plot(budgets, metric_avgs, line_style, color=color)

plt.legend()
plt.xlabel('Iteration n')
plt.ylabel('Give way rate (%)')
plt.xticks(budgets)
plt.grid(alpha=0.5)
# plt.savefig("planner_giveway_rate.pdf", dpi=500, bbox_inches='tight')

# %%
# %%
"""
Decision time
"""
plt.figure()
plt.xscale('log', basex=2)
kpi = 'max_decision_time'
for planner_name, planner_label, color, line_style in zip(planner_names, planner_labels, colors, line_styles):
    # if planner_name in ['mcts', 'belief_search']:
    if planner_name in ['omniscient', 'belief_search']:
        metrics = metric_logs[planner_name]
        metric = get_metric(metrics, kpi, planner_name)
        metric_avgs = [_metric.mean() for _metric in metric]
        plt.plot(budgets, metric_avgs, line_style, color=color, label=planner_label)

plt.xlabel('Iteration n')
plt.ylabel('Average Decision Time (s)')
plt.xticks(budgets)
plt.grid(alpha=0.5)
# plt.savefig("avg_decision_time.pdf", dpi=500, bbox_inches='tight')
