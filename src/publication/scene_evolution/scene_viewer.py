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

        self.end_merge_box = [Rectangle((config['merge_zone_end']+40, 0), \
                            200, config['lane_width'])]
        self.ramp_box = [Polygon(np.array([[config['merge_zone_end']-30, 0],
                                             [config['merge_zone_end']+40, config['lane_width']],
                                             [config['merge_zone_end']+40, 0]]))]
        self.ramp_bound = [Rectangle((0, config['lane_width']-0.25), \
                            config['merge_lane_start'], 0.5)]

        self.logged_var = {}

    def draw_initial_traffi_scene(self):
        self.fig = plt.figure(figsize=(10, 2))
        self.env_ax = self.fig.add_subplot(111)

        self.env_ax.set_xlim(0, self.config['lane_length'])
        self.env_ax.set_yticks([1, 3, 5, 7])
        self.env_ax.set_xlabel(r'Longitudinal position (m)')
        self.env_ax.set_ylabel(r'Lat. pos. (m)')




    def set_up_fig(self, subplot_count=3):
        if subplot_count == 3:
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

        elif subplot_count == 2:
            self.fig = plt.figure(figsize=(7, 5))
            self.speed_ax = self.fig.add_subplot(211)
            self.lateral_pos_ax = self.fig.add_subplot(212)
            for ax in self.fig.axes:
                ax.grid(alpha=0.6)
                # ax.set_xlim(-0.2, 6)
                ax.set_xlabel(r'Time (s)')

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
        ax.hlines(self.config['lane_width'], self.config['merge_lane_start'], self.config['merge_zone_end']+40,
                                colors='k', linestyles='--', linewidth=3)



    def draw_vehicle_belief(self, belief_info, max_depth, id):
        ax = self.env_ax
        # colors = cm.BuPu(np.linspace(1, 0, max_depth))
        colors = cm.rainbow(np.linspace(1, 0, max_depth))

        for depth, c in enumerate(colors):
            if depth > max_depth:
                continue
            ax.scatter(belief_info[id][depth]['xs'],
                       belief_info[id][depth]['ys'],
                       color=c, s=20, marker='x')

    def draw_vehicle(self, logged_states, id, time_step):
        ax = self.env_ax
        vehicle_colors = {'sdv':'green', 1:'orange', 2:'navy', 3:'blueviolet', 4:'blue', 5:'orchid'}
        logged_state = logged_states[id][logged_states[id][:, 0] == time_step][0]
        glob_x = logged_state[-2]
        glob_y = logged_state[-1]
        color = vehicle_colors[id]
        ax.scatter(glob_x, glob_y, s=100, marker=">", color=color, edgecolors=color)
        if id == 'sdv':
            ax.annotate('id:e', (glob_x-20, glob_y+0.7), color=color, size=12)
        else:
            ax.annotate('id:'+str(id), (glob_x-20, glob_y+0.7), color=color, size=12)
        ax.annotate('vel:'+str(round(logged_state[2], 1)), (glob_x-30, glob_y-1.3), color=color, size=12)


    def draw_sdv_traj(self, logged_states, time_step):
        ax = self.env_ax
        logged_state = logged_states[logged_states[:, 0] >= time_step]
        xs = logged_state[:, -2]
        ys = logged_state[:, -1]
        ax.plot(xs, ys, color='green', linewidth=2)
        ax.scatter(xs[::10], ys[::10], marker='>', color='black', s=50)

    def draw_scene(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax)

    def draw_speed(self, logged_states, color):
        speeds = logged_states[:, 2]
        time_axis = np.arange(len(speeds))/10
        self.speed_ax.plot(time_axis, speeds, color, linewidth=2)
        self.speed_ax.set_xlim(0, time_axis[-1]+0.5)
        self.speed_ax.set_ylabel(r'Long. speed (m/s)')

    def draw_lat_pos(self, logged_states):
        lat_poses = logged_states[:, -1]
        time_axis = np.arange(len(lat_poses))/10

        self.lateral_pos_ax.plot(time_axis, lat_poses, color='green', linewidth=2)
        self.lateral_pos_ax.set_ylabel(r'Lat. pos. (m)')
        self.lateral_pos_ax.set_ylim(-0.1, self.config['lane_width']*2)
        self.lateral_pos_ax.set_xlim(0, time_axis[-1]+0.5)
        self.lateral_pos_ax.set_yticks([1, 3, 5, 7])
