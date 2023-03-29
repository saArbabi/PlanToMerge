import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
import dill
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
        self.fig = plt.figure(figsize=(7, 2))
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
                ax.grid(alpha=0.3)
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
                ax.grid(alpha=0.3)
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


    def add_custom_legend(self):
        ax = self.env_ax
        legend_elements = [Line2D([0], [0], marker='>', color='w', label='${v_e}$',
                          markerfacecolor='g', markersize=15),
                            Line2D([0], [0], marker='>', color='w', label='${v_3}$',
                                              markerfacecolor='red', markersize=15),
                            Line2D([0], [0], marker='>', color='w', label='${v_4}$',
                                              markerfacecolor='blue', markersize=15)]

        # Create the figure
        ax.legend(handles=legend_elements, loc='lower left', facecolor='white')

    def draw_vehicle_belief(self, belief_info, max_depth_vis, id):
        ax = self.env_ax
        # colors = cm.BuPu(np.linspace(1, 0, max_depth))
        max_depth = 10

        colors = cm.rainbow(np.linspace(1, (1-(max_depth_vis/max_depth)), max_depth_vis))

        for depth, c in enumerate(colors):
            if depth > max_depth_vis:
                continue
            ax.scatter(belief_info[id][depth]['xs'],
                       belief_info[id][depth]['ys'],
                       color=c, s=50, alpha=0.3)

    def draw_vehicle_with_info(self, logged_states, id, time_step):
        ax = self.env_ax
        vehicle_colors = {'sdv':'green', 1:'orange', 2:'navy', 3:'red', 4:'blue', 5:'orchid'}
        logged_state = logged_states[id][logged_states[id][:, 0] == time_step][0]
        glob_x = logged_state[-2]
        glob_y = logged_state[-1]
        color = vehicle_colors[id]
        ax.scatter(glob_x, glob_y, s=150, marker=">", color=color,
                                        edgecolors='black', linewidth=1)
        if id == 'sdv':
            id = 'e'

        vehicle_id = 'v_' + str(id)
        ax.annotate('${}$'.format(vehicle_id), (glob_x-10, glob_y+0.7), color=color, size=14)

        if id == 4:
            ax.annotate('vel:'+str(round(logged_state[2], 1)), (glob_x-40, glob_y-1.3), color=color, size=12)
        else:
            ax.annotate('vel:'+str(round(logged_state[2], 1)), (glob_x-15, glob_y-1.3), color=color, size=12)

    def draw_vehicle(self, logged_states, id, time_step):
        ax = self.env_ax
        vehicle_colors = {'sdv':'green', 1:'orange', 2:'navy', 3:'red', 4:'blue', 5:'orchid'}
        logged_state = logged_states[id][logged_states[id][:, 0] == time_step][0]
        glob_x = logged_state[-2]
        glob_y = logged_state[-1]
        color = vehicle_colors[id]
        ax.scatter(glob_x, glob_y, s=150, marker=">", color=color,
                                        edgecolors='black', linewidth=1)

    def draw_sdv_traj(self, logged_states, max_depth_vis, time_step):
        ax = self.env_ax
        logged_state = logged_states[logged_states[:, 0] >= time_step]
        xs = logged_state[:, -2][::10].tolist() + [logged_state[:, -2][-1]]
        ys = logged_state[:, -1][::10].tolist() + [logged_state[:, -1][-1]]
        # xs = logged_state[:, -2][::10]
        # ys = logged_state[:, -1][::10]

        ax.plot(xs[1:], ys[1:], color='green', linewidth=12, alpha=0.8, zorder=1)

        max_depth = 10
        colors = cm.rainbow(np.linspace(1, (1-(max_depth_vis/max_depth)), len(xs)))


        for depth, c in enumerate(colors):
            if depth > 0:
                ax.scatter(xs[depth], ys[depth], color=c, s=60, edgecolor='black', zorder=2)

    def draw_scene(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax)

    def draw_speed(self, logged_states, color):
        speeds = logged_states[:, 2]
        time_axis = np.arange(len(speeds))/10
        self.speed_ax.plot(time_axis, speeds, color, linewidth=1.5)
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
