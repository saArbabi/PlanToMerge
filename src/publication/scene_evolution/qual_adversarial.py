"""
The following results are showcased:
>>>>>>>> comparing agent behaviour when dealing with a aggressive vs timid driver
Run plan_vis.py to get the search tree/traffic states.
episode: 511
Note:
150 itr allocated for planner budget.
Initiate the episode, vehicle 3 starts timid but then changed to aggressiveness
at time step 35.

"""
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from importlib import reload
import json
from src.publication.scene_evolution.scene_viewer import Viewer
%matplotlib tk

# from matplotlib.lines import Line2D

# %%
"""
Load logs
"""
# st = [
#      env.time_step,
#      vehicle.act_long_p,
#      vehicle.speed,
#      vehicle.glob_x,
#      vehicle.glob_y]

timestr = '20221125-03-06-37' # dealing with timid car
save_to = './src/publication/scene_evolution/saved_files/'
file_name = f'{timestr}_logged_states'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    logged_states = pickle.load(handle)
    logged_states['sdv'] = np.array(logged_states['sdv'])
    completion_index = np.where(logged_states['sdv'][1:, -1]-logged_states['sdv'][:-1, -1] != 0)[0][-1]
    logged_states['sdv'] = logged_states['sdv'][:completion_index+10, :]

    for car_id, state_vals in logged_states.items():
        if car_id != 'sdv':
            logged_states[car_id] = np.array(state_vals)
            logged_states[car_id] = logged_states[car_id][:logged_states['sdv'].shape[0], :]
#s %%

params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# %%


fig = plt.figure(figsize=(8     , 5))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.37)

for ax in fig.axes:
    ax.grid(alpha=0.6)
    # ax.set_xlim(-0.2, 6)
    ax.set_xlabel(r'Time (s)')


time_axis = np.arange(logged_states['sdv'].shape[0])/10
decision_times = [0, 10, 20, 30, 40, 90] # u u m m g g g g g m m m m
# decision_times_mid = [5, 15, 25, 30, 40, 100, 158] # up, up, up, idle, 2 x merge, give, merge
#################################################################################
delta_glob_x = np.abs(logged_states['sdv'][:, -2]-logged_states[3][:, -2])
ax_2.plot(time_axis, delta_glob_x, color='green', label='LVT')
ax_2.set_ylabel(r'Rel. long. distance (m)')
ax_2.set_xlim(-0.1, 15)
ax_2.set_ylim(-5, 110)
#################################################################################

decision_xs = np.array(decision_times)/10
decision_ys = delta_glob_x[decision_times]
text_ys = delta_glob_x[[5, 15, 25, 35, 65, 95, 120]]
#
for d in range(len(decision_times)):
    if d == 1:
        ax_2.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>', label='Decision')
    elif d == 5:
        # ax_2.scatter(, s=100, color='green', marker='>', rotate=30)
        ax_2.plot(decision_xs[d], decision_ys[d], marker=(3, 0, 80), markersize=13, color='green')

    else:
        ax_2.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>')
#
ax_2.text(decision_xs[1]+0.3, decision_ys[1]+32, 'increase ACC.', color='green', size=14)

ax_2.scatter(decision_xs[1], decision_ys[1]+35, color='k', s=20)

ax_2.plot([decision_xs[0]+0.5, decision_xs[1]], [text_ys[0], text_ys[1]+35],\
                                                                color='k', linewidth=1)
ax_2.plot([decision_xs[1]+0.5, decision_xs[1]], [text_ys[1], text_ys[1]+35],\
                                                                color='k', linewidth=1)
# ###############################################################################
#
ax_2.text(decision_xs[3]+0.3, text_ys[3]+15, 'merge-in', color='green', size=14)
ax_2.scatter(decision_xs[3], decision_ys[3]+17, color='k', s=20)
ax_2.plot([time_axis[25], time_axis[30]], [delta_glob_x[25], delta_glob_x[30]+17],\
                                                                color='k', linewidth=1)
ax_2.plot([time_axis[35], time_axis[30]], [delta_glob_x[35], delta_glob_x[30]+17],\
                                                                color='k', linewidth=1)

# ###############################################################################

ax_2.text(time_axis[60], delta_glob_x[60]-13, 'give-way', rotation=-27, color='green', size=14)
ax_2.text(time_axis[110], delta_glob_x[110]+12, 'merge-in', rotation=23, color='green', size=14)

ax_2.legend(ncol=1, edgecolor='black', facecolor='white')
#################################################################################

ax_1.plot(time_axis, logged_states[3][:, 2], color='red', label='Vehicle 3')
ax_1.plot(time_axis, logged_states['sdv'][:, 2], color='green', label='LVT')
ax_1.set_ylabel(r'Long. speed (m/s)')
ax_1.set_xlim(-0.1, 15)
ax_1.legend(ncol=1, edgecolor='black', facecolor='white')
#################################################################################
################################################################################
plt.savefig("adversarial.pdf", dpi=500, bbox_inches='tight')
