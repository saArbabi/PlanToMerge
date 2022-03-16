import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(10, 5))
        self.env_ax = self.fig.add_subplot(211)
        self.decision_ax = self.fig.add_subplot(212)
        self.focus_on_this_vehicle = None
        self.merge_box = [Rectangle((config['merge_lane_start'], 0), \
                            config['merge_lane_length'], config['lane_width'])]

    def draw_road(self, ax):
        lane_cor = self.config['lane_width']*self.config['lanes_n']
        ax.hlines(0, 0, self.config['lane_length'], colors='k', linestyles='solid')
        ax.vlines(self.config['merge_zone_end'], 0, self.config['lane_width'], \
                                                    colors='k', linestyles='solid')

        ax.vlines(self.config['ramp_exit_start'], 0, self.config['lane_width'], \
                                                    colors='k', linestyles='solid')

        ax.hlines(lane_cor, 0, self.config['lane_length'],
                                                    colors='k', linestyles='solid')

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(self.merge_box, hatch='/', alpha=0.2)
        ax.add_collection(pc)
        if self.config['lanes_n'] > 1:
            lane_cor = self.config['lane_width']
            for lane in range(self.config['lanes_n']-1):
                ax.hlines(lane_cor, 0, self.config['lane_length'],
                                        colors='k', linestyles='--')
                lane_cor += self.config['lane_width']

        ax.set_xlim(0, self.config['lane_length'])
        ax.set_yticks([])

    def draw_vehicles(self, ax, vehicles):
        glob_xs = [veh.glob_x for veh in vehicles]
        glob_ys = [veh.glob_y for veh in vehicles]

        annotation_mark_1 = [veh.id for veh in vehicles]
        annotation_mark_2 = [round(veh.speed, 1) for veh in vehicles]
        for i in range(len(annotation_mark_1)):
            ax.annotate(annotation_mark_1[i], (glob_xs[i], glob_ys[i]+1))
            ax.annotate(annotation_mark_2[i], (glob_xs[i], glob_ys[i]-1))

        for vehicle in vehicles:
            if str(vehicle.id) == self.focus_on_this_vehicle:
                print('#############  ', vehicle.id, '  ##############')
                print('My neighbours: ')
                for key, neighbour in vehicle.neighbours.items():
                    if neighbour:
                        print(key+': ', neighbour.id)
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                [vehicle.glob_y, neighbour.glob_y], linestyle='-',
                                    color='black', linewidth=1, alpha=0.3)
                    else:
                        print(key+': ', None)

                print('ego_decision: ', vehicle.lane_decision)
                print('ego_lane_id: ', vehicle.lane_id)
                print('ego_lane_id_target: ', vehicle.target_lane)
                print('glob_y: ', vehicle.glob_y)
                print('glob_x: ', round(vehicle.glob_x, 2))
                print('ego_act: ', vehicle.act_long)
                print('steps_since_lc_initiation: ', vehicle.steps_since_lc_initiation)
                print('driver_params: ', vehicle.driver_params)
                print('###########################')

                if vehicle.neighbours['f']:
                    print('delta_x: ', vehicle.neighbours['f'].glob_x - vehicle.glob_x)

            if 'att' in vehicle.neighbours:
                neighbour = vehicle.neighbours['att']
                if neighbour:
                    line_1 = [vehicle.glob_y, neighbour.glob_y+.6]
                    line_2 = [vehicle.glob_y, neighbour.glob_y-.6]
                    ax.fill_between([vehicle.glob_x, neighbour.glob_x+1], \
                                        line_1, line_2, alpha=0.3, color='grey')


            if vehicle.lane_decision == 'move_left':
                ax.scatter(vehicle.glob_x-2, vehicle.glob_y+.7,
                                s=50, marker="*", color='red', edgecolors='black')
            elif vehicle.lane_decision == 'move_right':
                ax.scatter(vehicle.glob_x-2, vehicle.glob_y-.7,
                                s=50, marker="*", color='red', edgecolors='black')


        color_shade = [veh.driver_params['aggressiveness'] for veh in vehicles]
        ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        c=color_shade, cmap='rainbow')

    def draw_attention_line(self, ax, vehicles):
        x1 = vehicles[0].x
        y1 = vehicles[0].y
        x2 = vehicles[0].attend_veh.glob_x
        y2 = vehicles[0].attend_veh.glob_y
        ax.plot([x1, x2],[y1, y2])
        ax.scatter([x1, x2],[y1, y2], s=10)

    def draw_highway(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax)
        self.draw_vehicles(ax, vehicles)
        # self.draw_attention_line(ax, vehicles)

    def draw_beliefs(self, ax, sdv):
        if not sdv.planner.belief_info:
            return

        belief_info = sdv.planner.belief_info
        max_depth = len(belief_info)
        colors = cm.rainbow(np.linspace(1, 0, max_depth))

        for depth, c in enumerate(colors):
            ax.scatter(belief_info[depth]['xs'],
                       belief_info[depth]['ys'],
                       color=c, s=5)

            # ax.annotate(str(len(belief_info[depth]['xs'])), \
            #             (belief_info[depth]['xs'][-1], belief_info[depth]['ys'][-1]))

    def draw_plans(self, ax, sdv):
        if not sdv.planner.tree_info:
            return

        for plan_itr in sdv.planner.tree_info:
            ax.plot(plan_itr['x_rollout'], plan_itr['y_rollout'], 'o-', \
                                        markersize=3, alpha=0.5, color='orange')
            ax.plot(plan_itr['x'], plan_itr['y'], '-o', \
                                        markersize=3, alpha=0.2, color='black')

    def draw_decision_counts(self, ax, sdv):
        if not sdv.decisions_and_counts:
            return
        ax.clear()
        decisions = sdv.decisions_and_counts['decisions']
        counts = sdv.decisions_and_counts['counts']
        decisions_and_counts = [x for x in sorted(zip(decisions, counts))]

        for decision, count in decisions_and_counts:
            if count == max(counts):
                max_count = count
                color = 'green'
            else:
                color = 'grey'

            ax.annotate(count, (decision, count/2))

            ax.bar(decision, count, 0.5, \
                    label=sdv.OPTIONS[decision][1], color=color)
        ax.set_ylim([0, max_count+1])

        ax.set_xticks(list(sdv.OPTIONS.keys()))
        ax.set_xticklabels(['LANEKEEP \n TIMID',
                            'LANEKEEP \n NORMAL',
                            'LANEKEEP \n AGGRESSIVE',
                            'MERGE \n TIMID',
                            'MERGE \n NORMAL',
                            'MERGE \n AGGRESSIVE'])

    def render(self, vehicles, sdv):
        self.draw_highway(self.env_ax, vehicles)
        # self.draw_decision_counts(self.decision_ax, sdv)
        # self.draw_plans(self.env_ax, sdv)
        # self.draw_beliefs(self.env_ax, sdv)

        self.fig.tight_layout()
        plt.pause(1e-10)
