"""
The following results are showcased:
>>>>>>>> comparing agent behaviour when dealing with a aggressive vs normal driver
Run plan_vis.py to get the search tree/traffic states.
episode: 511
Note:
150 itr allocated for planner budget.
Initiate the episode, but modify vehicle position and aggressiveness:

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
# %%

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from importlib import reload
import json
from scene_viewer import Viewer

# from matplotlib.lines import Line2D

# %%
"""
Load logs
"""
plot_cat = 'aggressive'
# plot_cat = 'normal'
if plot_cat == 'aggressive':
    timestr = '20221125-03-58-53' # dealing with aggressive car
    time_step = 85
elif plot_cat == 'normal':
    timestr = '20221125-03-47-18' # dealing with normal car
    time_step = 25


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

logged_states['sdv'].shape
logged_states[2].shape
# %%
"""
Plot initial scene
"""
# %matplotlib tk
# from src.publication.scene_evolution import scene_viewer
# reload(scene_viewer)
# from src.publication.scene_evolution.scene_viewer import Viewer


with open('../../envs/config.json', 'rb') as handle:
    config = json.load(handle)

plot_viewer = Viewer(config)
plot_viewer.draw_initial_traffi_scene()

time_step = 6
plot_viewer.draw_road()
plot_viewer.draw_vehicle_with_info(logged_states, id='sdv', time_step=time_step)

plot_viewer.draw_vehicle_with_info(logged_states, id=4, time_step=time_step)
plot_viewer.draw_vehicle_with_info(logged_states, id=3, time_step=time_step)
plot_viewer.draw_vehicle_with_info(logged_states, id=2, time_step=time_step)
plot_viewer.draw_vehicle_with_info(logged_states, id=1, time_step=time_step)
# plt.savefig("scene_plot_initial.pdf", dpi=500, bbox_inches='tight')


# %%
"""
Plot scene
"""
 
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
import scene_viewer
reload(scene_viewer)
from scene_viewer import Viewer

with open('../../envs/config.json', 'rb') as handle:
    config = json.load(handle)

plot_viewer = Viewer(config)
plot_viewer.set_up_fig()
plot_viewer.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.35)
plot_viewer.draw_road()
plot_viewer.draw_vehicle(logged_states, id='sdv', time_step=time_step)

if plot_cat == 'aggressive':
    max_depth_vis = 6
    plot_viewer.draw_sdv_traj(logged_states['sdv'], max_depth_vis+3, time_step=time_step)
    plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis+3, id=3)
    plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis, id=4)
    plot_viewer.draw_vehicle(logged_states, id=4, time_step=time_step)
    plot_viewer.draw_vehicle(logged_states, id=3, time_step=time_step)
    ############
    plot_viewer.draw_speed(logged_states['sdv'], 'green')
    plot_viewer.draw_speed(logged_states[3], 'red')
    plot_viewer.draw_speed(logged_states[4], 'blue')
    plot_viewer.draw_lat_pos(logged_states['sdv'])

elif plot_cat == 'normal':
    max_depth_vis = 10
    plot_viewer.draw_sdv_traj(logged_states['sdv'], max_depth_vis, time_step=time_step)
    plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis, id=3)
    plot_viewer.draw_vehicle_belief(belief_info, max_depth_vis-7, id=4)
    plot_viewer.draw_vehicle(logged_states, id=4, time_step=time_step)
    plot_viewer.draw_vehicle(logged_states, id=3, time_step=time_step)
    ############
    plot_viewer.draw_speed(logged_states['sdv'], 'green')
    plot_viewer.draw_speed(logged_states[3], 'red')
    plot_viewer.draw_speed(logged_states[4], 'blue')
    plot_viewer.draw_lat_pos(logged_states['sdv'])
plot_viewer.speed_ax.set_ylim(0, 26)
plot_viewer.speed_ax.legend(['${v_e}$', '${v_3}$', '${v_4}$'],
                    loc='lower right' ,ncol=1, edgecolor='black', facecolor='white')
plot_viewer.lateral_pos_ax.legend(['${v_e}$'], loc='upper left' , ncol=1, edgecolor='black', facecolor='white')
#s %%
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

im = image.imread('./time_color_bar.png')
imagebox = OffsetImage(im, zoom=0.1)
ab = AnchoredOffsetbox(loc=1, bbox_to_anchor=(450, 553), child=imagebox, frameon=False)
plot_viewer.env_ax.add_artist(ab)

plot_viewer.add_custom_legend()
# plot_viewer.fig.figimage(im, 0, plot_viewer.fig.bbox.ymax-10)

if plot_cat == 'aggressive':
    plt.savefig("scene_plot_aggressive.pdf", dpi=500, bbox_inches='tight')
elif plot_cat == 'normal':
    plt.savefig("scene_plot_normal.pdf", dpi=500, bbox_inches='tight', facecolor='w')




# %%
