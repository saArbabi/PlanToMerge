from envs.auto_merge import EnvAutoMerge
import time
from datetime import datetime
import os
import numpy as np
import pickle
import json
import tensorflow as tf

class MCEVAL():
    def __init__(self, eval_config):
        self.env = EnvAutoMerge()
        self.eval_config = eval_config

    def create_empty(self):
        """
        Create empty files in which monte carlo logs can be dumped.
        """
        self.mc_collection = {}

    def update_eval_config(self, exp_name):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.eval_config['mc_config']
        progress_logging = self.eval_config['progress_logging'][exp_name]
        progress_logging['last_update'] = dt_string
        progress_logging['current_episode_count'] = \
                                    f'{self.current_episode_count}/{mc_config["episodes_n"]}'

        progress_logging['episode_in_prog'] = self.episode_id

        if self.current_episode_count == mc_config['episodes_n']:
            self.eval_config['status'] = 'COMPLETE'
        else:
            self.eval_config['status'] = 'IN PROGRESS ...'

        self.eval_config['progress_logging'][exp_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.eval_config, f, ensure_ascii=False, indent=4)

    def load_planner(self, planner_info, planner_name):
        if planner_name == 'qmdp':
            from tree_search.qmdp import QMDP
            self.planner = QMDP(planner_info)
        elif planner_name == 'mcts':
            from tree_search.mcts import MCTSDPW
            self.planner = MCTSDPW(planner_info)
        elif planner_name == 'omniscient':
            from tree_search.omniscient import Omniscient
            self.planner = Omniscient(planner_info)
        else:
            print('No such planner exists yet. Check the evaluation config file')

    def run_episode(self, episode_id):
        print('Running episode: ', episode_id)
        self.current_episode_count += 1
        self.env = EnvAutoMerge()
        self.env.initialize_env(episode_id)
        self.planner._enough_history = False
        self.planner.steps_till_next_decision = 0
        self.planner.seed(2022)
        tf.random.set_seed(2022)


        cumulative_decision_count = 0
        decision_times = []
        cumulative_reward = 0
        hard_brake_count = 0
        decisions_made = []

        while not self.env.sdv.is_merge_initiated():
            if self.planner.is_decision_time():
                t_0 = time.time()
                self.planner.plan(self.env)
                decision = self.planner.get_decision(self.env)
                t_1 = time.time()

                self.env.sdv.update_decision(decision)
                cumulative_decision_count += 1
                decision_times.append(t_1 - t_0)
                cumulative_reward += self.env.get_reward(decision)
                decisions_made.append(decision)
                if self.env.got_bad_action:
                    hard_brake_count += 1
                self.env.env_reward_reset()

            self.env.step()
            self.planner.steps_till_next_decision -= 1

        cumulative_reward += self.env.get_reward(decision)
        # collect metrics
        timesteps_to_merge = self.env.time_step
        max_decision_time = max(decision_times)
        self.mc_collection[episode_id] = [
                                        cumulative_reward,
                                        timesteps_to_merge,
                                        max_decision_time,
                                        hard_brake_count,
                                        decisions_made]

    def initiate_eval(self, planner_name):
        self.current_episode_count = 0
        self.create_empty()
        progress_logging = {}
        self.episode_id = 500
        self.target_episode = self.eval_config['mc_config']['episodes_n'] + \
                                                            self.episode_id

        progress_logging['episode_in_prog'] = self.episode_id
        progress_logging['current_episode_count'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.eval_config['progress_logging'][planner_name] = progress_logging

    def load_collections(self, exp_name, planner_name):
        exp_dir = './src/evaluation/experiments/'+planner_name
        with open(exp_dir+'/'+exp_name+'.pickle', 'rb') as handle:
            self.mc_collection = pickle.load(handle)

    def dump_mc_logs(self, exp_name, planner_name):
        exp_dir = './src/evaluation/experiments/'+planner_name
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        with open(exp_dir+'/'+exp_name+'.pickle', 'wb') as handle:
            pickle.dump(self.mc_collection, handle)
        print('################## Dumping mc_logs: ', exp_name)

    def is_eval_complete(self, exp_name, planner_name):
        """Check if this planner has been fully evaluated.
        """
        if not exp_name in self.eval_config['progress_logging']:
            self.initiate_eval(exp_name)
            return False

        progress_logging = self.eval_config['progress_logging'][exp_name]
        mc_config = self.eval_config['mc_config']
        epis_n_left = 0 # remaining episodes ot compelte

        current_episode_count = progress_logging['current_episode_count']
        current_episode_count = current_episode_count.split('/')
        self.current_episode_count = int(current_episode_count[0])
        epis_n_left = mc_config['episodes_n'] - self.current_episode_count
        if epis_n_left == 0:
            return True
        else:
            self.load_collections(exp_name, planner_name)
            self.episode_id = progress_logging['episode_in_prog'] + 1
            progress_logging['current_episode_count'] = \
                        f'{self.current_episode_count}/{mc_config["episodes_n"]}'
            self.target_episode =  self.episode_id + epis_n_left
            self.update_eval_config(exp_name)
            return False

    def run(self):
        planner_info = self.eval_config['planner_info']
        planner_names = planner_info['planner_names']
        budgets = planner_info['budgets']
        for planner_name in planner_names:
            for budget in budgets:
                exp_name = f'{planner_name}-budget-{budget}'
                if self.is_eval_complete(exp_name, planner_name):
                    continue
                planner_info['budget'] = budget
                self.load_planner(planner_info, planner_name)
                while self.episode_id < self.target_episode:
                    self.run_episode(self.episode_id)
                    self.dump_mc_logs(exp_name, planner_name)
                    self.update_eval_config(exp_name)
                    self.episode_id += 1
