from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
from tree_search.mcts import MCTSDPW
import numpy as np

import json

class SDVehicle(IDMMOBILVehicleMerge):
    OPTIONS = {0: ['LANEKEEP', 'TIMID'],
               1: ['LANEKEEP', 'NORMAL'],
               2: ['LANEKEEP', 'AGGRESSIVE'],
               3: ['MERGE', 'TIMID'],
               4: ['MERGE', 'NORMAL'],
               5: ['MERGE', 'AGGRESSIVE']}

    def __init__(self, id):
        self.id = id
        self.planner = MCTSDPW()
        self.actme = [0, 0]
        self.budget = 10 # timesteps with 0.1s step size
        self.decisions_and_counts = None
        self.decision = None
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)
            self.merge_lane_start = config['merge_lane_start']
            self.ramp_exit_start = config['ramp_exit_start']

    def get_sdv_decision(self, env_state, obs):
        # if self.time_lapse % self.budget == 0:
        if self.time_lapse % self.budget == 0:
            self.planner.plan(env_state, obs)
            decision, self.decisions_and_counts = self.planner.get_decision()
            # self.decision = 0
            return decision
        return self.decision

    def get_driver_param(self, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act']:
            # the larger the param, the more aggressive the driver
            min_value = self.parameter_range['least_aggressvie'][param_name]
            max_value = self.parameter_range['most_aggressive'][param_name]
            return  min_value + self.driver_params['aggressiveness']*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness',
                                                'act_threshold', 'safe_braking']:
            # the larger the param, the more timid the driver
            min_value = self.parameter_range['most_aggressive'][param_name]
            max_value = self.parameter_range['least_aggressvie'][param_name]
            return  max_value - self.driver_params['aggressiveness']*(max_value-min_value)

    def set_driver_params(self):
        self.driver_params['desired_v'] = self.get_driver_param('desired_v')
        self.driver_params['desired_tgap'] = self.get_driver_param('desired_tgap')
        self.driver_params['min_jamx'] = self.get_driver_param('min_jamx')
        self.driver_params['max_act'] = self.get_driver_param('max_act')
        self.driver_params['min_act'] = self.get_driver_param('min_act')

    def act(self, decision):
        # print(self.driver_params['aggressiveness'])
        if not self.decision or self.decision != decision:
            self.decision = decision
            merge_decision = self.OPTIONS[decision][0]
            driving_style = self.OPTIONS[decision][1]

            if driving_style == 'TIMID':
                self.driver_params['aggressiveness'] = 0
            elif driving_style == 'NORMAL':
                self.driver_params['aggressiveness'] = 0.5
            elif driving_style == 'AGGRESSIVE':
                self.driver_params['aggressiveness'] = 1
            self.set_driver_params()

            if merge_decision == 'MERGE':
                self.lane_decision = 'move_left'
            elif merge_decision == 'LANEKEEP':
                self.lane_decision = 'keep_lane'

        # print(self.neighbours['f'])
        act_long = self.idm_action(self, self.neighbours['att'])
        # act_lat = self.lateral_action()
        if self.lane_decision == 'move_left':
            act_lat = 1
        elif self.lane_decision == 'keep_lane':
            act_lat = 0

        return [act_long, act_lat]
        # return [0, 0]

    def planner_observe(self):
        """Observation used by the planner
        """
        delta_x_to_merge = self.ramp_exit_start-self.glob_x
        return self.glob_y
        # return self.glob_y + np.random.NORMAL()
        # return delta_x_to_merge
