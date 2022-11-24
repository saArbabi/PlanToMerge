import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import time
import pickle

class Viewer():
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['GIVEWAY', 'IDLE'],
               6 : ['ABORT', 'IDLE']
               }

    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(6, 6))
        self.env_ax = self.fig.add_subplot(311)
        self.decision_ax = self.fig.add_subplot(312)
        self.var_ax = self.fig.add_subplot(313)
        self.focus_on_this_vehicle = None
        self.merge_box = [Rectangle((config['merge_lane_start'], 0), \
                            config['merge_lane_length'], config['lane_width'])]
        self.logged_states = {}
        self.timestr = time.strftime("%Y%m%d-%H-%M-%S") # time the experiment is run

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
                print('ego_act: ', vehicle.act_long_c)
                print('driver_params: ', vehicle.driver_params)
                if vehicle.neighbours['rl']:
                    print('delta_rl-x: ', vehicle.glob_x-vehicle.neighbours['rl'].glob_x)
                print('###########################')

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

    def draw_beliefs(self, ax, planner):
        if not planner.belief_info:
            return
        belief_info = planner.belief_info
        max_depth = len(belief_info)
        colors = cm.rainbow(np.linspace(1, 0, max_depth))

        for depth, c in enumerate(colors):
            ax.scatter(belief_info[depth]['xs'],
                       belief_info[depth]['ys'],
                       color=c, s=5)

            # ax.annotate(str(len(belief_info[depth]['xs'])), \
            #             (belief_info[depth]['xs'][-1], belief_info[depth]['ys'][-1]))

    def draw_plans(self, ax, planner):
        if not planner.tree_info:
            return

        longest_tree_length = 0
        for i, plan_itr in enumerate(planner.tree_info):
            ax.plot(plan_itr['x_rollout'], plan_itr['y_rollout'], 'o-', \
                                        markersize=3, alpha=0.2, color='orange')

            ax.plot(plan_itr['x'], plan_itr['y'], '-o', \
                                        markersize=3, alpha=0.2, color='black')

            if len(plan_itr['x']) > longest_tree_length:
                longest_tree_length = len(plan_itr['x'])
                longest_tree_index = i

        longest_plan = planner.tree_info[longest_tree_index]
        ax.scatter(longest_plan['x'][-1], \
                   longest_plan['y'][-1], color='red', marker='|', s=1000)

    def draw_decision_counts(self, ax, planner):
        if not planner.decision_counts:
            return
        ax.clear()
        decisions = planner.decision_counts['decisions']
        counts = planner.decision_counts['counts']
        decisions_and_counts = [x for x in sorted(zip(decisions, counts))]

        for decision, count in decisions_and_counts:
            if count == max(counts):
                max_count = count
                color = 'green'
            else:
                color = 'grey'

            ax.annotate(count, (decision, count/2))

            ax.bar(decision, count, 0.5, \
                    label=self.OPTIONS[decision][1], color=color)
        ax.set_ylim([0, max_count])
        ax.set_xlim([0, 7])

        ax.set_xticks(list(self.OPTIONS.keys()))
        ax.set_xticklabels(['LANEKEEP \n UP (1)',
                            'LANEKEEP \n IDLE (2)',
                            'LANEKEEP \n DOWN (3)',
                            'MERGE \n IDLE (4)',
                            'GIVEWAY \n IDLE (5)',
                             'ABORT \n IDLE (6)'])
    def fetch_state(self, env, vehicle):
        st = [
             env.time_step,
             vehicle.act_long_p,
             vehicle.speed,
             vehicle.glob_x,
             vehicle.glob_y]
        return st

    def log_state(self, env):
        if not self.logged_states:
            self.logged_states['sdv'] = []
            for veh in env.vehicles:
                self.logged_states[veh.id] = []

        self.logged_states['sdv'].append(self.fetch_state(env, env.sdv))

        for veh in env.vehicles:
            self.logged_states[veh.id].append(self.fetch_state(env, veh))

    def draw_var(self, ax):
        """
        Plots long. actions for agent and one other vehicle
        """
        ax.clear()
        o_name = 'Yielder_'+str(self.draw_var_veh.id)
        sdv_sts = self.logged_states['sdv']
        sdv_acts = [sdv_st[1] for sdv_st in sdv_sts]
        o_sts = self.logged_states[self.draw_var_veh.id]
        o_acts = [o_st[1] for o_st in o_sts]
        log_len = len(sdv_acts)
        if log_len < 150:
            ax.plot(sdv_acts[-150:], color='red', label='Merger')
            ax.plot(o_acts[-150:], color='blue', label=o_name)
            ax.scatter(np.arange(0, len(o_acts), 10), \
                       o_acts[::10], color='blue', label=o_name)
            ax.set_xlim(0, 180)
        else:
            ax.plot(sdv_acts, color='red', label='Merger')
            ax.plot(o_acts, color='blue', label=o_name)
            ax.scatter(np.arange(0, len(o_acts), 10), \
                       o_acts[::10], color='blue', label=o_name)
            ax.set_xlim(log_len-150, log_len+30)

        ax.set_ylim(-7, 7)
        ax.plot(200*[-4], color='red', linestyle='--')
        ax.set_ylabel('long.acc ($ms^{-2}$)')
        ax.set_xlabel('step')
        ax.legend()
        ax.grid()

    def render_plans(self, planner):
        if planner.decision_counts:
            self.draw_decision_counts(self.decision_ax, planner)
            self.draw_plans(self.env_ax, planner)
            # self.draw_beliefs(self.env_ax, planner)
            plt.pause(1e-10)

    def render_logs(self, avg_step_reward_steps, avg_step_rewards):
        if self.draw_var_veh:
            self.draw_var(self.var_ax)
        for time_step, reward in zip(avg_step_reward_steps, avg_step_rewards):
            self.var_ax.text(time_step, 5.1, str(round(reward, 1)), fontsize='xx-small')

    def save_tree_state(self, planner, last_decision_time_step):
        """
        Use this for overlaying the search tree onto the road
        """
        save_to = './src/publication/scene_evolution/saved_files/'
        # tree_info:
        file_name = f'{self.timestr}_tree_info_step_{last_decision_time_step}'
        with open(save_to+file_name+'.pickle', 'wb') as handle:
            pickle.dump(planner.tree_info, handle)

        # belief_info
        file_name = f'{self.timestr}_belief_info_step_{last_decision_time_step}'
        with open(save_to+file_name+'.pickle', 'wb') as handle:
            pickle.dump(planner.belief_info, handle)

        print('Tree state saved: ' + file_name)


    def save_state_logs(self):
        """
        Use this for plotting vehicle states
        Return:
        log: {id:[[time_step, act_long, act_lat, speed, glob_x, glob_y], ...], ...}
        """
        save_to = './src/publication/scene_evolution/saved_files/'
        file_name = f'{self.timestr}_logged_states'
        with open(save_to+file_name+'.pickle', 'wb') as handle:
            pickle.dump(self.logged_states, handle)

        print('Logged states saved: ' + file_name)

    def save_latent(self, planner):
        save_to = './src/publication/scene_evolution/saved_files/'
        file_name = f'{self.timestr}_saved_latent'
        with open(save_to+file_name+'.pickle', 'wb') as handle:
            pickle.dump(planner.root.state.hidden_state, handle)
        print('Latent vector saved: ' + file_name)

    def render_env(self, vehicles):
        self.draw_highway(self.env_ax, vehicles)
        self.fig.tight_layout()
        plt.pause(1e-10)
