import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
import dill

class Viewer():
    def __init__(self, config):
        self.config  = config
        # self.fig = plt.figure(figsize=(6, 2))

        self.end_merge_box = [Rectangle((config['merge_zone_end'], 0), \
                            200, config['lane_width'])]
        self.ramp_box = [Polygon(np.array([[config['merge_zone_end']-30, 0],
                                             [config['merge_zone_end'], config['lane_width']],
                                             [config['merge_zone_end'], 0]]))]
        self.ramp_bound = [Rectangle((0, config['lane_width']-0.25), \
                            config['merge_lane_start'], 0.5)]

        self.logged_var = {}

    def set_up_fig(self):
        self.fig = plt.figure(figsize=(7, 9))
        self.env_ax = self.fig.add_subplot(311)
        self.speed_ax = self.fig.add_subplot(312)
        self.lateral_pos_ax = self.fig.add_subplot(313)
        for ax in self.fig.axes[1:]:
            ax.grid(alpha=0.6)
            # ax.set_xlim(-0.2, 6)
            ax.set_xlabel(r'Time (s)')

        self.env_ax.set_xlim(0, self.config['lane_length'])
        self.env_ax.set_yticks([1, 3, 5, 7])
        self.env_ax.set_xlabel(r'Longitudinal position (m)')
        self.env_ax.set_ylabel(r'Lat. pos. (m)')

    def draw_road(self):
        ax = self.env_ax
        lane_cor = self.config['lane_width']*self.config['lanes_n']
        ax.hlines(0, 0, self.config['lane_length'], colors='k', linestyles='solid')

        ax.hlines(lane_cor, 0, self.config['lane_length'],
                                                    colors='k', linestyles='solid')

        # Create patch collection with specified colour/alpha
        pc_end = PatchCollection(self.end_merge_box, hatch='/', alpha=0.3, color='k')
        pc_ramp = PatchCollection(self.ramp_box, hatch='/', alpha=0.3, color='k')
        pc_ramp_bound = PatchCollection(self.ramp_bound, hatch='/', alpha=0.3, color='k')
        ax.add_collection(pc_end)
        ax.add_collection(pc_ramp)
        ax.add_collection(pc_ramp_bound)
        ax.hlines(self.config['lane_width'], self.config['merge_lane_start'], self.config['merge_zone_end'],
                                colors='k', linestyles='--', linewidth=3)



    def draw_vehicle_belief(self, belief_info, id):
        ax = self.env_ax
        max_depth = len(belief_info[id])
        # colors = cm.BuPu(np.linspace(1, 0, max_depth))
        colors = cm.rainbow(np.linspace(1, 0, max_depth))

        for depth, c in enumerate(colors):
            if depth == 0:
                continue
            ax.scatter(belief_info[id][depth]['xs'],
                       belief_info[id][depth]['ys'],
                       color=c, s=30, alpha=0.5)

    def draw_vehicle(self, logged_states, id, time_step):
        ax = self.env_ax
        vehicle_colors = {'sdv':'red', 1:'green', 2:'navy', 3:'blueviolet', 4:'blue', 5:'orchid'}
        logged_states = np.array(logged_states[id])
        logged_state = logged_states[logged_states[:, 0] == time_step][0]
        glob_x = logged_state[-2]
        glob_y = logged_state[-1]
        color = vehicle_colors[id]
        ax.scatter(glob_x, glob_y, s=100, marker=">", color=color, edgecolors=color)
        if id == 'sdv':
            ax.annotate('id:e', (glob_x-20, glob_y+0.7), color=color)
        else:
            ax.annotate('id:'+str(id), (glob_x-20, glob_y+0.7), color=color)
        ax.annotate('vel:'+str(round(logged_state[2], 1)), (glob_x-30, glob_y-1.5), color=color)


    def draw_sdv_traj(self, logged_states, time_step):
        ax = self.env_ax
        logged_states = np.array(logged_states)
        logged_state = logged_states[logged_states[:, 0] >= time_step]
        xs = logged_state[:, -2]
        ys = logged_state[:, -1]
        ax.plot(xs, ys, color='red')
        ax.scatter(xs[::10], ys[::10], marker='>', color='black', s=50)

    def draw_ego_plan(self, tree_info):
        ax = self.env_ax
        longest_tree_length = 0
        for i, plan_itr in enumerate(tree_info):
            if i > 0:
                break
            ax.plot(plan_itr['x_rollout'], plan_itr['y_rollout'], 'o-', \
                                        markersize=3, alpha=0.2, color='orange')

            ax.plot(plan_itr['x'], plan_itr['y'], '-o', \
                                        markersize=3, alpha=0.2, color='black')

            if len(plan_itr['x']) > longest_tree_length:
                longest_tree_length = len(plan_itr['x'])
                longest_tree_index = i

        longest_plan = tree_info[longest_tree_index]
        ax.scatter(longest_plan['x'][-1], \
                   longest_plan['y'][-1], color='red', marker='|', s=1000)

    def draw_scene(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax)

    def draw_speed(self, logged_states, color):
        speeds = [logged_state[2] for logged_state in logged_states]
        self.speed_ax.plot(np.arange(len(speeds))/10, speeds, color)
        self.speed_ax.set_ylabel(r'Long. speed (m/s)')

    def draw_lat_pos(self, logged_states):
        lat_poses = [logged_state[-1] for logged_state in logged_states]
        self.lateral_pos_ax.plot(np.arange(len(lat_poses))/10, lat_poses, color='red')
        self.lateral_pos_ax.set_ylabel(r'Lat. pos. (m)')

    def draw_state_profiles(self):
        time_steps = np.linspace(0, 5.9, 60)
        traj_len = len(self.trace_log['mveh']['speed'])
        lw = 2.5

        self.speed_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['speed'], color='red', linestyle='--', linewidth=lw)
        self.speed_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['speed'], color='green', linestyle='-', linewidth=lw)
        self.speed_ax.set_ylabel(r'Long. speed (m/s)')
        self.speed_ax.yaxis.set_ticks(np.arange(10., 13, 1))
        self.speed_ax.set_ylim(10, 12.5)

        self.lateral_pos_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['act_long'], color='red', linestyle='--', linewidth=lw)
        self.lateral_pos_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['act_long'], color='green', linestyle='-', linewidth=lw)
        self.lateral_pos_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.lateral_pos_ax.set_ylabel(r'Long. Accel. $\mathdefault{(m/s^2)}$')

        self.act_lat_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['act_lat'], color='red', linestyle='--', linewidth=lw)
        self.act_lat_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['act_lat'], color='green', linestyle='-', linewidth=lw)
        self.act_lat_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.act_lat_ax.set_ylabel(r'Lateral speed (m/s)')
