"""
The following results are showcased:
>>>>>>>> comparing agent behaviour when dealing with a aggressive vs timid driver
Run plan_vis.py to get the search tree/traffic states.
episode: 511
Note:
100 itr allocated for planner budget.
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
# %matplotlib tk

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

timestr = '20221027-20-46-55' # dealing with timid car
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
fig = plt.figure(figsize=(8     , 5))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.37)

for ax in fig.axes:
    ax.grid(alpha=0.6)
    # ax.set_xlim(-0.2, 6)
    ax.set_xlabel(r'Time (s)')


time_axis = np.arange(logged_states['sdv'].shape[0])/10
decision_times = [0, 10, 20, 30, 40, 50, 60, 100] # up, up, up, idle, 2 x merge, give, merge
# decision_times_mid = [5, 15, 25, 30, 40, 100, 158] # up, up, up, idle, 2 x merge, give, merge
#################################################################################
delta_glob_x = np.abs(logged_states['sdv'][:, -2]-logged_states[3][:, -2])
ax_2.plot(time_axis, delta_glob_x, color='green', label='LVT')
ax_2.set_ylabel(r'Rel. long. distance (m)')
ax_2.set_xlim(-0.1, 16.5)
ax_2.set_ylim(-5, 110)
#################################################################################

decision_xs = np.array(decision_times)/10
decision_ys = delta_glob_x[decision_times]
text_ys = delta_glob_x[[5, 15, 25, 35, 45, 55, 80, 120]]
#
for d in range(len(decision_times)):
    if d == 1:
        ax_2.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>', label='Decision')
    elif d == 6:
        # ax_2.scatter(, s=100, color='green', marker='>', rotate=30)
        ax_2.plot(decision_xs[d], decision_ys[d], marker=(3, 0, 0), markersize=13, color='green')
    elif d == 7:
        # ax_2.scatter(, s=100, color='green', marker='>', rotate=30)
        ax_2.plot(decision_xs[d], decision_ys[d], marker=(3, 0, 80), markersize=13, color='green')

    else:
        ax_2.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>')

ax_2.text(decision_xs[1]+0.7, decision_ys[1]+21, 'increase ACC setpoint', color='green', size=11)
ax_2.scatter(decision_xs[1]+0.5, decision_ys[1]+20, color='k', s=20)
ax_2.plot([decision_xs[2]+0.5, decision_xs[1]+0.5], [text_ys[2], text_ys[1]+20],\
                                                                color='k', linewidth=1)
ax_2.plot([decision_xs[0]+0.5, decision_xs[1]+0.5], [text_ys[0], text_ys[1]+20],\
                                                                color='k', linewidth=1)
ax_2.plot([decision_xs[1]+0.5, decision_xs[1]+0.5], [text_ys[1], text_ys[1]+20],\
                                                                color='k', linewidth=1)


ax_2.text(decision_xs[3]+0.7, text_ys[3]-18, 'maintain', color='green', size=11)
ax_2.scatter(decision_xs[3]+0.5, decision_ys[3]-16, color='k', s=20)
ax_2.plot([decision_xs[3]+0.5, decision_xs[3]+0.5], [text_ys[3], text_ys[2]-15],\
                                                                color='k', linewidth=1)

text_x = 5.5
text_y = decision_ys[1]+10
ax_2.text(text_x, text_y, 'merge in', color='green', size=11)
ax_2.scatter(text_x-0.2, text_y, color='k', s=20)

ax_2.plot([decision_xs[4]+0.5, text_x-0.2], [text_ys[4], text_y],\
                                                                color='k', linewidth=1)
ax_2.plot([decision_xs[5]+0.5, text_x-0.2], [text_ys[5], text_y],\
                                                                color='k', linewidth=1)
# ax_2.plot([decision_xs[0]+0.5, decision_xs[1]+0.5], [text_ys[0], text_ys[1]+20],\
#                                                                 color='k', linewidth=1)
# ax_2.plot([decision_xs[1]+0.5, decision_xs[1]+0.5], [text_ys[1], text_ys[1]+20],\
#                                                                 color='k', linewidth=1)




# ax_2.scatter(decision_xs[1]+0.5, text_ys[1]+20, color='k', s=20)
#
ax_2.text(8, text_ys[6]-17, 'give way', rotation=-40, color='green', size=11)
ax_2.text(12, text_ys[7]+10, 'merge in', rotation=27, color='green', size=11)
#

ax_2.legend(ncol=1, edgecolor='black', facecolor='white')
#################################################################################

ax_1.plot(time_axis, logged_states[3][:, 2], color='purple', label='Vehicle 3')
ax_1.plot(time_axis, logged_states['sdv'][:, 2], color='green', label='LVT')
ax_1.set_ylabel(r'Long. speed (m/s)')
ax_1.set_xlim(-0.1, 16.5)
ax_1.legend(ncol=1, edgecolor='black', facecolor='white')

#################################################################################

#################################################################################



################################################################################
plt.savefig("adversarial.pdf", dpi=500, bbox_inches='tight')

params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
 
