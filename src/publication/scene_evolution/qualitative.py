"""
The following results are showcased:
1) comparing agent behaviour when dealing with a aggressive vs timid driver
2) comparing LVT with MCTS and QMDP(?)
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
timestr = '20220909-20-16-29'
save_to = './src/publication/scene_evolution/'
time_step = '26'
file_name = f'{timestr}_tree_info_step_{time_step}'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    tree_info = pickle.load(handle)

file_name = f'{timestr}_belief_info_step_{time_step}'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    belief_info = pickle.load(handle)

file_name = f'{timestr}_logged_states'
with open(save_to+file_name+'.pickle', 'rb') as handle:
    logged_states = pickle.load(handle)

# %%
"""
Plot scene
"""
from src.publication.scene_evolution import scene_viewer
reload(scene_viewer)
from src.publication.scene_evolution.scene_viewer import Viewer
plot_viewer = Viewer(config)


with open('./src/envs/config.json', 'rb') as handle:
    config = json.load(handle)

time_step = 15
plot_viewer.set_up_fig()
plot_viewer.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

plot_viewer.draw_road()
plot_viewer.draw_sdv_traj(logged_states['sdv'], time_step=time_step)
plot_viewer.draw_vehicle(logged_states, id='sdv', time_step=time_step)

plot_viewer.draw_vehicle_belief(belief_info, id=3)
belief_info[1]
plot_viewer.draw_vehicle(logged_states, id=3, time_step=time_step)

# plt.savefig("scene_plot.pdf", dpi=500, bbox_inches='tight')

# plot_viewer.draw_ego_plan(tree_info)




#s %%
"""
Plot state speed and lateral position
"""
plot_viewer.draw_speed(logged_states['sdv'], 'red')
plot_viewer.draw_speed(logged_states[3], 'purple')

plot_viewer.draw_lat_pos(logged_states['sdv'])

plot_viewer.speed_ax.legend(['Agent', 'Vehicle 3'],
                      ncol=1, edgecolor='black', facecolor='white')


#s %%
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 18,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 18
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

plt.savefig("scene_plot.pdf", dpi=500, bbox_inches='tight')




# %%
