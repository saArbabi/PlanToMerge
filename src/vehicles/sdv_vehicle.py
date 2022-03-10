from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
from tree_search.mcts import MCTSDPW

import json

class SDVehicle(IDMMOBILVehicleMerge):
    OPTIONS = {0: ['LK', 'UP'],
                1: ['LK', 'DOWN'],
                2: ['LK', 'MERGE']}

    def __init__(self, id):
        self.id = id
        self.planner = MCTSDPW()
        self.actme = [0, 0]
        self.budget = 10 # timesteps with 0.1s step size
        self.decisions_and_counts = None
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)
            self.merge_lane_start = config['merge_lane_start']
            self.ramp_exit_start = config['ramp_exit_start']

    def get_sdv_decision(self, env_state, obs):
        # if self.time_lapse % self.budget == 0:
        if self.time_lapse % self.budget == 0:
            self.planner.plan(env_state, obs)
            self.decision, self.decisions_and_counts = self.planner.get_decision()
            print(self.decisions_and_counts)
            # self.decision = 0
        return self.decision

    def act(self, decision):
        if self.OPTIONS[decision][1] == 'UP':
            act_long = 1
            act_lat = 0
        if self.OPTIONS[decision][1] == 'DOWN':
            act_long = -1
            act_lat = 0
        if self.OPTIONS[decision][1] == 'MERGE':
            act_long = 0
            act_lat = 1
        return [act_long, act_lat]

    def observe(self):
        delta_x_to_merge = self.ramp_exit_start-self.glob_x
        return delta_x_to_merge
