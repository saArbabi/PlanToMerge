from envs.auto_merge import EnvAutoMerge
import time
from datetime import datetime
import os
import numpy as np
import pickle
import json

class MCEVAL():
    def __init__(self, eval_config):
        self.env = EnvAutoMerge()
        self.eval_config = eval_config

    def create_empty(self):
        """
        Create empty files in which monte carlo logs can be dumped.
        """
        self.mc_collection = {}

    def update_eval_config(self, planner_name):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.eval_config['mc_config']
        progress_logging = self.eval_config['progress_logging'][planner_name]
        progress_logging['last_update'] = dt_string
        progress_logging['current_episode_count'] = \
                                    f'{self.current_episode_count}/{mc_config["episodes_n"]}'

        progress_logging['episode_in_prog'] = self.episode_id

        if self.current_episode_count == mc_config['episodes_n']:
            self.eval_config['status'] = 'COMPLETE'
        else:
            self.eval_config['status'] = 'IN PROGRESS ...'

        self.eval_config['progress_logging'][planner_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.eval_config, f, ensure_ascii=False, indent=4)

    def load_planner(self, planner_name):
        if planner_name == 'qmdp':
            from tree_search.qmdp import QMDP
            self.planner = QMDP()
        elif planner_name == 'mcts':
            from tree_search.mcts import MCTSDPW
            self.planner = MCTSDPW()
        else:
            print('No such planner exists yet. Check the evaluation config file')


    def dump_mc_logs(self, planner_name):
        exp_dir = './src/evaluation/experiments/'+planner_name
        with open(exp_dir+'/mc_collection.pickle', 'wb') as handle:
            pickle.dump(self.mc_collection, handle)

    def run_episode(self, episode_id):
        print('Running episode: ', episode_id)
        self.current_episode_count += 1
        self.env.initialize_env(episode_id)

        cumulative_decision_count = 0
        cumulative_decision_time = 0
        hard_brake_count = 0

        while not self.env.sdv.is_merge_complete():
        # while self.env.time_step < 40:
            if self.planner.is_decision_time():
                t_0 = time.time()
                self.planner.plan(self.env)
                decision = self.planner.get_decision()
                t_1 = time.time()
                cumulative_decision_count += 1
                cumulative_decision_time += (t_1 - t_0)

            else:
                decision = self.env.sdv.decision

            for vehicle in self.env.vehicles:
                if vehicle.act_long_c and vehicle.act_long_c < -5:
                    hard_brake_count += 1

            self.env.step(decision)
            self.planner.steps_till_next_decision -= 1
            print(self.env.time_step)

        # collect metrics
        timesteps_to_merge = self.env.time_step
        time_per_decision = cumulative_decision_time/cumulative_decision_count
        self.mc_collection[episode_id] = [
                                        timesteps_to_merge,
                                        time_per_decision,
                                        hard_brake_count]

    def initiate_eval(self, planner_name):
        self.current_episode_count = 0
        self.create_empty()
        progress_logging = {}
        self.episode_id = 3
        self.target_episode = self.eval_config['mc_config']['episodes_n'] + \
                                                            self.episode_id

        progress_logging['episode_in_prog'] = self.episode_id
        progress_logging['current_episode_count'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.eval_config['progress_logging'][planner_name] = progress_logging

    def load_collections(self, planner_name):
        exp_dir = './src/evaluation/experiments/'+ planner_name
        with open(exp_dir+'/mc_collection.pickle', 'rb') as handle:
            self.mc_collection = pickle.load(handle)

    def is_eval_complete(self, planner_name):
        """Check if this planner has been fully evaluated.
        """
        if not planner_name in self.eval_config['progress_logging']:
            self.initiate_eval(planner_name)
            return False

        progress_logging = self.eval_config['progress_logging'][planner_name]
        mc_config = self.eval_config['mc_config']
        epis_n_left = 0 # remaining episodes ot compelte

        current_episode_count = progress_logging['current_episode_count']
        current_episode_count = current_episode_count.split('/')
        self.current_episode_count = int(current_episode_count[0])
        epis_n_left = mc_config['episodes_n'] - self.current_episode_count
        if epis_n_left == 0:
            return True
        else:
            self.load_collections(planner_name)
            self.episode_id = progress_logging['episode_in_prog']
            progress_logging['current_episode_count'] = \
                        f'{self.current_episode_count}/{mc_config["episodes_n"]}'
            self.target_episode =  self.episode_id + epis_n_left
            self.update_eval_config(planner_name)
            return False

    def run(self):
        planner_names = self.eval_config['planner_names']
        for planner_name in planner_names:
            if self.is_eval_complete(planner_name):
                continue
            self.load_planner(planner_name)
            print('Planner being evaluated: ', planner_name)
            while self.episode_id < self.target_episode:
                self.run_episode(self.episode_id)
                self.dump_mc_logs(planner_name)
                self.update_eval_config(planner_name)
                self.episode_id += 1
