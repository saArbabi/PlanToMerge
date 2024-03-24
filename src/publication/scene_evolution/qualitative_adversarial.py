"""
The following results are showcased:
>>>>>>>> comparing agent behaviour when dealing with a aggressive vs timid driver
Run planner_vis.py to get the search tree/traffic states.
episode: 511
Note:
150 itr allocated for planner budget.
Initiate the episode, vehicle 3 starts timid but then changed to aggressiveness
at time step 35.
tips with planner_vis: 
1) input 'st' at timestep you want to save tree states. These are later used for visualising
beliefs. 
2) input 'slogs' at the end of the episode so that full trajecotires are logged. 

"""

# %%

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from importlib import reload
import json
from scene_viewer import Viewer
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
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

timestr = '20221125-03-06-37' # dealing with timid car
save_to = '/home/salar/my_projects/PlanToMerge/PlanToMerge/src/publication/scene_evolution/saved_files/'
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
# a%%

params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# %%



#################################################################################
# %%
"""
Load logs
"""
import matplotlib.image as image

timestr = '20240323-15-31-30' # dealing with aggressive car
time_step = 95


save_to = '/home/salar/my_projects/PlanToMerge/PlanToMerge/src/publication/scene_evolution/saved_files/'
file_name = f'{timestr}_tree_info_step_{time_step}'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    tree_info = pickle.load(handle)

file_name = f'{timestr}_belief_info_step_{time_step}'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    belief_info = pickle.load(handle)

file_name = f'{timestr}_logged_states'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    logged_states = pickle.load(handle)
    logged_states['sdv'] = np.array(logged_states['sdv'])
    logged_states['sdv'] = logged_states['sdv'][logged_states['sdv'][:, -2] <= 405]

    for car_id, state_vals in logged_states.items():
        if car_id != 'sdv':
            logged_states[car_id] = np.array(state_vals)
            logged_states[car_id] = logged_states[car_id][:logged_states['sdv'].shape[0], :]


with open('../../envs/config.json', 'rb') as handle:
    config = json.load(handle)
# s%%
class ViewerAdversary(Viewer):
    def __init__(self, config):
        super().__init__(config)

    def set_up_fig(self):
        self.fig = plt.figure(figsize=(7, 9))
        self.env_ax = self.fig.add_subplot(311)
        self.speed_ax = self.fig.add_subplot(312)
        self.delta_glob_x = self.fig.add_subplot(313)
        for ax in self.fig.axes[1:]:
            ax.grid(alpha=0.3)
            # ax.set_xlim(-0.2, 6)
            ax.set_xlabel(r'Time (s)')

        self.env_ax.set_xlim(0, self.config['lane_length'])
        self.env_ax.set_yticks([1, 3, 5, 7])
        self.env_ax.set_xlabel(r'Longitudinal position (m)')
        self.env_ax.set_ylabel(r'Lat. pos. (m)')
 
 
plot_viewer = ViewerAdversary(config)
plot_viewer.set_up_fig()
plot_viewer.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.35)
plot_viewer.draw_road()
plot_viewer.draw_vehicle(logged_states, id='sdv', time_step=time_step)
############
max_depth_vis = 6
plot_viewer.draw_sdv_traj(logged_states['sdv'], max_depth_vis+3, time_step=time_step)
plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis-1, id=4)
plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis+3, id=3)
plot_viewer.draw_vehicle(logged_states, id=4, time_step=time_step)
plot_viewer.draw_vehicle(logged_states, id=3, time_step=time_step)
#################################################################################
im = image.imread('./time_color_bar.png')
imagebox = OffsetImage(im, zoom=0.1)
ab = AnchoredOffsetbox(loc=1, bbox_to_anchor=(455, 553), child=imagebox, frameon=False)
plot_viewer.env_ax.add_artist(ab)

plot_viewer.add_custom_legend()
 
#################################################################################
ax_2 = plot_viewer.speed_ax
ax_3 = plot_viewer.delta_glob_x

time_axis = np.arange(logged_states['sdv'].shape[0])/10
decision_times = [0, 10, 20, 30, 40, 90] # u u m m g g g g g m m m m
# decision_times_mid = [5, 15, 25, 30, 40, 100, 158] # up, up, up, idle, 2 x merge, give, merge
#################################################################################
delta_glob_x = np.abs(logged_states['sdv'][:, -2]-logged_states[3][:, -2])
ax_3.plot(time_axis, delta_glob_x, color='green', label=r'$v_e$')
ax_3.set_ylabel(r'Rel. long. distance (m)')
ax_3.set_xlim(-0.1, 15)
ax_3.set_ylim(-5, 110)
#################################################################################

decision_xs = np.array(decision_times)/10
decision_ys = delta_glob_x[decision_times]
text_ys = delta_glob_x[[5, 15, 25, 35, 65, 95, 120]]
#
for d in range(len(decision_times)):
    if d == 1:
        ax_3.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>', label='Decision')
    elif d == 5:
        # ax_3.scatter(, s=100, color='green', marker='>', rotate=30)
        ax_3.plot(decision_xs[d], decision_ys[d], marker=(3, 0, 80), markersize=13, color='green')

    else:
        ax_3.scatter(decision_xs[d], decision_ys[d], s=100, color='green', marker='>')
#
ax_3.text(decision_xs[1]+0.3, decision_ys[1]+32, 'increase ACC.', color='green', size=14)

ax_3.scatter(decision_xs[1], decision_ys[1]+35, color='k', s=20)

ax_3.plot([decision_xs[0]+0.5, decision_xs[1]], [text_ys[0], text_ys[1]+35],\
                                                                color='k', linewidth=1)
ax_3.plot([decision_xs[1]+0.5, decision_xs[1]], [text_ys[1], text_ys[1]+35],\
                                                                color='k', linewidth=1)
# ###############################################################################
#
ax_3.text(decision_xs[3]+0.3, text_ys[3]+20, 'merge-in', color='green', size=14)
ax_3.scatter(decision_xs[3], decision_ys[3]+17, color='k', s=20)
ax_3.plot([time_axis[25], time_axis[30]], [delta_glob_x[25], delta_glob_x[30]+17],\
                                                                color='k', linewidth=1)
ax_3.plot([time_axis[35], time_axis[30]], [delta_glob_x[35], delta_glob_x[30]+17],\
                                                                color='k', linewidth=1)

# ###############################################################################

ax_3.text(time_axis[60] - 1, delta_glob_x[60] - 2, 'give-way', rotation=-27, color='green', size=14)
ax_3.text(time_axis[110] - 1, delta_glob_x[110] - 1, 'merge-in', rotation=30, color='green', size=14)

ax_3.legend(ncol=1, edgecolor='black', facecolor='white')
#################################################################################

ax_2.plot(time_axis, logged_states['sdv'][:, 2], color='green', label='${v_e}$')
ax_2.plot(time_axis, logged_states[3][:, 2], color='red', label='${v_3}$')
ax_2.set_ylabel(r'Long. speed (m/s)')
ax_2.set_xlim(-0.1, 15)
ax_2.legend(ncol=1, edgecolor='black', facecolor='white')
#################################################################################
################################################################################
plt.savefig("adversarial.pdf", dpi=500, bbox_inches='tight')
