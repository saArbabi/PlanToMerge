from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
import numpy as np

import json

class SDVehicle(IDMMOBILVehicleMerge):
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'UP'],
               5 : ['MERGE', 'IDLE'],
               6 : ['MERGE', 'DOWN']}


    def __init__(self, id):
        self.id = id
        self.decision_steps_n = 10 # timesteps with 0.1s step size
        self.decisions_and_counts = None
        self.decision = None
        self.decision_cat = 'LANEKEEP'
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)
            self.merge_lane_start = config['merge_lane_start']
            self.ramp_exit_start = config['ramp_exit_start']

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

    # def change_driving_style(self, driving_style):
    #     aggressiveness = self.driver_params['aggressiveness']
    #     if driving_style == 'UP':
    #         aggressiveness = min(0.9, aggressiveness + 0.1)
    #     elif driving_style == 'IDLE':
    #         pass
    #     elif driving_style == 'DOWN':
    #         aggressiveness = max(0.1, aggressiveness - 0.1)
    #     return aggressiveness

    def change_driving_style(self, driving_style):
        aggressiveness = self.driver_params['aggressiveness']
        if driving_style == 'UP':
            aggressiveness = 0.6
        elif driving_style == 'IDLE':
            aggressiveness = 0.5
        elif driving_style == 'DOWN':
            aggressiveness = 0.4
        return aggressiveness

    def is_merge_possible(self):
        if self.glob_x > self.merge_lane_start:
            return True

    def act(self, decision):
        # print(self.driver_params['aggressiveness'])
        if not self.decision or self.decision != decision:
            merge_decision = self.OPTIONS[decision][0]
            driving_style = self.OPTIONS[decision][1]
            self.decision = decision
            self.decision_cat = merge_decision
            # print('decision ', decision)
            # print('aggressiveness ', self.driver_params['aggressiveness'])

            self.driver_params['aggressiveness'] = self.change_driving_style(driving_style)
            self.set_driver_params()

            if merge_decision == 'MERGE':
                self.lane_decision = 'move_left'

            elif merge_decision == 'LANEKEEP' or merge_decision == 'ABORT':
                self.lane_decision = 'keep_lane'
                if self.target_lane != self.lane_id:
                    self.target_lane = self.lane_id

        # if self.decision_cat == 'ABORT' and not self.neighbours['att']:
        #     if self.speed > 0:
        #         act_long = -1
        #     if self.speed <= 0:
        #         act_long = 0
        #         self.speed = 0
        #
        # else:
        #     act_long = self.idm_action(self, self.neighbours['att'])

        act_long = self.idm_action(self, self.neighbours['att'])

        if self.is_merge_complete():
            if self.neighbours['rl']:
                self.neighbours['rl'].neighbours['f'] = self
                self.neighbours['rl'].neighbours['m'] = None
            self.lane_decision = 'keep_lane'
            self.glob_y = 1.5*self.lane_width

        act_lat = self.lateral_action()

        # act_rl_lc = self.idm_action(self.neighbours['rl'], self)
        # if self.neighbours['rl']:
        #     print(self.neighbours['rl'].id, ' ', act_rl_lc, ' ', (self.lane_decision), ' ', self.glob_x)
        # else:
        #     print('no rl car ', (self.lane_decision), ' ', self.glob_x)
        # print('sdv glob_x ', self.glob_x)
        # print('sdv lane id ', self.lane_id)

        # if self.lane_decision == 'move_left':
        #     act_lat = 1
        # elif self.lane_decision == 'keep_lane':
        #     act_lat = 0

        return [act_long, act_lat]
        # return [0, 0]
