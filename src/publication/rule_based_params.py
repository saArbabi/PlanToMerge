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
          'legend.fontsize': 12,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


# %%
""" Load sim collections
"""

run_name = 'rule_based_param_test'
planner_names = [
                'rule_based_10',
                'rule_based_20',
                'rule_based_30',
                'rule_based_40',
                'rule_based_50',
                'rule_based_60'
                ]


hard_brakes_logs = {}
give_way_logs = {}
for planner_name in planner_names:
    hard_brakes_logs[planner_name] = {}
    give_way_logs[planner_name] = {}
    exp_dir = './src/evaluation/experiments/'+run_name+'/'+planner_name
    mc_collection = get_mc_collection(exp_dir, ['rule_based-budget-1.pickle'])
    for episode, epis_metric in mc_collection[0].items():
        # if episode > 509:
        #     continue
        hard_brakes_logs[planner_name][episode] = epis_metric[-3]
        give_way_logs[planner_name][episode] = epis_metric[-2].count(5)/(epis_metric[-2].count(5)+epis_metric[-2].count(4))


hard_brakes_logs[planner_name].values()
give_way_logs[planner_name].values()
epis_metric[-2]
# %%
threshod_percentile = [10, 20, 30, 40, 50, 60]
hard_brakes_avgs = []
hard_brakes_standard_error = []
for planner_name in planner_names:
    hard_brakes_vals = np.array(list((hard_brakes_logs[planner_name].values())))
    hard_brakes_avgs.append(hard_brakes_vals.mean())
    hard_brakes_standard_error.append(hard_brakes_vals.std()/np.sqrt(100))

give_way_avgs = []
give_way_standard_error = []
for planner_name in planner_names:
    give_way_vals = np.array(list((give_way_logs[planner_name].values())))
    give_way_avgs.append(give_way_vals.mean())
    give_way_standard_error.append(give_way_vals.std()/np.sqrt(100))

# %%
plt.figure(figsize=(6, 2))
ax = plt.subplot(111)
# plt.figure(figsize=(6, 3))
ax.errorbar(threshod_percentile, hard_brakes_avgs, hard_brakes_standard_error, color='black', capsize=3, linestyle='None')
ax.plot(threshod_percentile, hard_brakes_avgs, 'o--', color='black', label='Hard brakes', markersize=5)

ax.errorbar(threshod_percentile, give_way_avgs, hard_brakes_standard_error, color='black', capsize=3, linestyle='None')
ax.plot(threshod_percentile, give_way_avgs, 's-', color='black', label='Give way', markersize=5)

# plt.ylim(-0.05, 1.05)
ax.set_xticks(threshod_percentile)
ax.set_xlabel('Threshold percentile')
ax.grid(alpha=0.2)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig("param_testing.pdf", dpi=500, bbox_inches='tight')
