import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
import os
import pickle
import sys
import json
import tensorflow as tf
from importlib import reload
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, './src')

def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    """ Returns the state sequence. It also deletes the middle index, which is
        the transition point from history to future.
    """
    # data shape: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], history_len-1, axis=1)
    return traj.flatten()

def latent_samples(model, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    merger_cs = future_m_veh_c[sample_index, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
    z_, _ = model.belief_net.sample_z(latent_dis_param)
    return z_


hf_usc_indexs = {}

col_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'm_veh_exists', 'e_veh_att',
         'e_veh_action_p', 'f_veh_action_p', 'm_veh_action_p',
         'e_veh_action_c', 'f_veh_action_c', 'm_veh_action_c',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'aggressiveness',
         'desired_v','desired_tgap', 'min_jamx', 'max_act', 'min_act',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
         'em_delta_y', 'delta_x_to_merge']
for i, item_name in enumerate(col_names):
    hf_usc_indexs[item_name] = i
# %%
"""
Load logs
"""
plot_files = {'scenario_1':['20220920-03-50-11'], \
                 'scenario_2':['20220920-04-11-20']
                  }
plot_cats = {'scenario_1':'NA', 'scenario_2':'NA'}
save_to = './src/publication/scene_evolution/saved_files/'

for plot_name in plot_cats.keys():
    file_name = f'{plot_files[plot_name][0]}_saved_latent'
    with open(save_to+file_name+'.pickle', 'rb') as handle:
        plot_cats[plot_name] = pickle.load(handle)
plot_cats[plot_name][-1][-1].shape
len(plot_cats[plot_name][-1])
# %%
"""
Load data
"""
history_len = 30 # steps
rollout_len = 50
data_id = '049'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

data_files_dir = '../DriverActionEstimators/src/datasets/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    data_arrays = pickle.load(handle)
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_c = np.float32(future_m_veh_c)
history_future_usc.shape

all_epis = np.unique(history_sca[:, 0, 0])
np.random.seed(2021)
np.random.shuffle(all_epis)
train_epis = all_epis[:int(len(all_epis)*0.7)]
val_epis = np.setdiff1d(all_epis, train_epis)
train_samples = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_samples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
# %%
"""
Load model (with config file)
"""
model_name = 'neural_idm_367'
epoch_count = '20'
exp_path = '../DriverActionEstimators/src/models/experiments/'+model_name+'/model_epo'+epoch_count
exp_dir = os.path.dirname(exp_path)
with open(exp_dir+'/'+'config.json', 'rb') as handle:
    config = json.load(handle)
    print(json.dumps(config, ensure_ascii=False, indent=4))

from models.core import neural_idm
reload(neural_idm)
from models.core.neural_idm import NeurIDMModel
model = NeurIDMModel(config)
model.forward_sim.rollout_len = 50

model.load_weights(exp_path).expect_partial()

with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
    model.forward_sim.env_scaler = pickle.load(handle)

with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
    model.forward_sim.dummy_value_set = pickle.load(handle)
# %%
"""Prediction for episode: 511
"""
traces_n = 50
tf.random.set_seed(2021)
scenario_latents = {}
for plot_name in plot_cats.keys():
    latent_dis_param = plot_cats[plot_name][-1]
    latent_dis_param_v3 = [param[-2:-1, :] for param in latent_dis_param]
    latent_dis_param_v3 = [tf.repeat(param, traces_n, axis=0) for param in latent_dis_param_v3]
    z_idm, _ = model.belief_net.sample_z(latent_dis_param_v3)
    scenario_latents[plot_name] = z_idm

# %%
""" plot setup
"""
MEDIUM_SIZE = 14
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# %%
"""
2D Latent figure
"""
zsamples_n = 5000
examples_to_vis = np.random.choice(val_samples, zsamples_n, replace=False)
sampled_z = latent_samples(model, examples_to_vis).numpy()
# %%
fig = plt.figure(figsize=(6.5, 4))
ax = fig.add_subplot(111)
plot_name = 'scenario_1'

aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]
color_shade = aggressiveness
latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1],
                s=10, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
z_idm = scenario_latents[plot_name]
ax.grid(False)
ax.scatter(z_idm[:, 0], z_idm[:, 1], s=20, marker="x", edgecolors='none', color='black')
ax.scatter(z_idm[:, 0].numpy().mean(), z_idm[:, 1].numpy().mean(), s=5000, marker="s", edgecolors='black', facecolors='none')

axins = inset_axes(ax,
                    width="5%",
                    height="90%",
                    loc='right',
                    borderpad=-2
                   )
fig.colorbar(latent_plot, cax=axins, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
plt.ylabel('$\psi$', fontsize=25, rotation=0, labelpad=12)
ax.set_xlabel('$z_1$', fontsize=35)
ax.set_ylabel('$z_2$', fontsize=35)
# ax.set_xlim(-7, 5.5)
ax.set_ylim(-12.5, 12.5)
ax.minorticks_off()

plt.savefig("latent_scenario_1.jpg", dpi=500, bbox_inches='tight')
# %%
fig = plt.figure(figsize=(6.5, 4))
ax = fig.add_subplot(111)
plot_name = 'scenario_2'

aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]
color_shade = aggressiveness
latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1],
                s=10, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
z_idm = scenario_latents[plot_name]
ax.grid(False)
ax.scatter(z_idm[:, 0], z_idm[:, 1], s=20, marker="x", edgecolors='none', color='black')
ax.scatter(z_idm[:, 0].numpy().mean(), z_idm[:, 1].numpy().mean(), s=5000, marker="s", edgecolors='black', facecolors='none')

axins = inset_axes(ax,
                    width="5%",
                    height="90%",
                    loc='right',
                    borderpad=-2
                   )
fig.colorbar(latent_plot, cax=axins, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
plt.ylabel('$\psi$', fontsize=25, rotation=0, labelpad=12)
ax.set_xlabel('$z_1$', fontsize=35)
ax.set_ylabel('$z_2$', fontsize=35)
# ax.set_xlim(-7, 5.5)
ax.set_ylim(-12.5, 12.5)
ax.minorticks_off()

plt.savefig("latent_scenario_2.jpg", dpi=500, bbox_inches='tight')
