"""
The following results are showcased:
>>>>>> comparing LVT with MCTS and QMDP(?)
episode: 511
Note:
100 itr allocated for planner budget.
Initiate the episode, but modify vehicle position/gaggressiveness:

for vehicle in vehicles: # see env_initializor.py
    if vehicle.id == 3:
        # vehicle.driver_params['aggressiveness'] = 0.9
        vehicle.driver_params['aggressiveness'] = 0.1
        vehicle.glob_x -= 20
        vehicle.set_driver_params(self.rng)

    if vehicle.id == 4:
        vehicle.glob_x -= 40

return vehicles
"""
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from importlib import reload
import json
# from matplotlib.lines import Line2D

# %%
"""
Load logs
"""
plot_files = {'lvt':['20221027-21-18-04'], \
                 'qmdp':['20221109-01-47-07'],
                 'mcts':['20221027-21-13-12'],
                 'omniscient':['20221027-21-21-37'],
                 'mcts_mean':['20221109-02-05-36'],
                 'rule_based':['20221109-02-09-39'],
                  } # [time_stamp, time_step]

plot_cats = {'lvt':'NA', 'mcts':'NA', 'omniscient':'NA',
             'qmdp':'NA', 'mcts_mean':'NA', 'rule_based':'NA'}
save_to = './src/publication/scene_evolution/saved_files/'

for planner_name in plot_cats.keys():
    file_name = f'{plot_files[planner_name][0]}_logged_states'
    with open(save_to+file_name+'.pickle', 'rb') as handle:
        logged_states = pickle.load(handle)
        logged_states['sdv'] = np.array(logged_states['sdv'])
        completion_index = np.where(logged_states['sdv'][1:, -1]-logged_states['sdv'][:-1, -1] != 0)[0][-1]
        logged_states['sdv'] = logged_states['sdv'][:completion_index+10, :]

        for car_id, state_vals in logged_states.items():
            if car_id != 'sdv':
                logged_states[car_id] = np.array(state_vals)
                logged_states[car_id] = logged_states[car_id][:logged_states['sdv'].shape[0], :]

        plot_cats[planner_name] = logged_states

# %%
"""
Plot veh states
"""
# %matplotlib tk
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 10,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
from src.publication.scene_evolution import scene_viewer
reload(scene_viewer)
from src.publication.scene_evolution.scene_viewer import Viewer
with open('./src/envs/config.json', 'rb') as handle:
    config = json.load(handle)
plot_viewer = Viewer(config)

plot_viewer.set_up_fig(2)
plot_viewer.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.35)
colors = ['darkgreen', 'blue', 'red', 'darkgreen', 'blue', 'k']
line_styles = ['-', '-', '-', '--', '--', '--']
planner_labels = ['LVT', 'MCTS', 'Omniscient', 'QMDP', 'MCTS normal', 'Rule-based']

for planner_name, planner_label, color, line_style in zip(plot_cats.keys(), planner_labels, colors, line_styles):
    # speed
    speeds = plot_cats[planner_name]['sdv'][:, 2]
    time_axis = np.arange(len(speeds))/10
    plot_viewer.speed_ax.plot(time_axis, speeds, line_style, color=color, label=planner_label)


    # lat pos
    speeds = plot_cats[planner_name]['sdv'][:, -1]
    plot_viewer.lateral_pos_ax.plot(time_axis, speeds, line_style, color=color, label=planner_label)

    # logged_states = plot_cats[planner_name]
    # plot_viewer.draw_speed(logged_states['sdv'], 'red')
    # plot_viewer.draw_lat_pos(logged_states['sdv'])
    # plot_viewer.speed_ax.set_ylim(-1, 26)
plot_viewer.speed_ax.set_xlim(0, 18)
plot_viewer.speed_ax.set_ylim(0, 30)
plot_viewer.speed_ax.set_ylabel(r'Long. speed (m/s)')
plot_viewer.speed_ax.legend(edgecolor='black', facecolor='white')
# plot_viewer.speed_ax.legend(loc='upper right', ncol=3, edgecolor='black', facecolor='white')

plot_viewer.lateral_pos_ax.set_ylabel(r'Lat. pos. (m)')
plot_viewer.lateral_pos_ax.set_ylim(-0.1, config['lane_width']*2)
plot_viewer.lateral_pos_ax.set_xlim(0, 18)
plot_viewer.lateral_pos_ax.set_yticks([1, 3, 5, 7])
# plot_viewer.lateral_pos_ax.legend(ncol=1, edgecolor='black', facecolor='white')

#s %%


plt.savefig("qualitative_planner_comparison.pdf", dpi=500, bbox_inches='tight')




# %%
11.6-16.9
5.3/11.6
