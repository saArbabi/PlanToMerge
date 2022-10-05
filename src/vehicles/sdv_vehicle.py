from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
import numpy as np
import json

class SDVehicle(IDMMOBILVehicleMerge):
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['GIVEWAY', 'IDLE']
               }

    def __init__(self, id):
        self.id = id
        self.decisions_and_counts = None
        self.decision = 2
        self.abort_been_chosen = False
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
        # IDM params
        self.driver_params['desired_v'] = self.get_driver_param('desired_v')
        self.driver_params['desired_tgap'] = self.get_driver_param('desired_tgap')
        self.driver_params['min_jamx'] = self.get_driver_param('min_jamx')
        self.driver_params['max_act'] = self.get_driver_param('max_act')
        self.driver_params['min_act'] = self.get_driver_param('min_act')

    def change_aggressiveness(self, speed_decision):
        if speed_decision == 'UP':
            self.driver_params['aggressiveness'] = \
                                min(self.driver_params['aggressiveness']+0.3, 1)
        elif speed_decision == 'DOWN':
            self.driver_params['aggressiveness'] = \
                                max(self.driver_params['aggressiveness']-0.3, 0)
        self.set_driver_params()

    def is_merge_possible(self):
        if self.glob_x > self.merge_lane_start \
                    and not self.is_merge_complete():
            return True

    def is_merge_initiated(self):
        return self.glob_y > 0.5*self.lane_width

    def update_decision(self, decision):
        self.old_neighbours = self.neighbours
        self.decision = decision
        merge_decision = self.OPTIONS[decision][0]
        speed_decision = self.OPTIONS[decision][1]

        if speed_decision != 'IDLE':
            self.change_aggressiveness(speed_decision)

        if merge_decision == 'MERGE':
            self.lane_decision = 'move_left'

        elif merge_decision in ['LANEKEEP', 'ABORT', 'GIVEWAY']:
            self.lane_decision = 'keep_lane'
            if self.target_lane != self.lane_id:
                self.target_lane = self.lane_id

    def act(self):
        act_long = self.idm_action(self, self.neighbours['att'])

        if self.is_merge_complete():
            if self.neighbours['rl']:
                self.neighbours['rl'].neighbours['f'] = self
                self.neighbours['rl'].neighbours['m'] = None
            self.lane_decision = 'keep_lane'
            self.glob_y = 1.5*self.lane_width

        act_lat = self.lateral_action()

        return [act_long, act_lat]
