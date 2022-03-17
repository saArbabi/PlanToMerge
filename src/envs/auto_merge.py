from envs.merge import EnvMerge
from vehicles.sdv_vehicle import SDVehicle
import numpy as np
import copy
from envs.env_initializor_test import EnvInitializor
# from envs.env_initializor import EnvInitializor
import sys

class EnvAutoMerge(EnvMerge):
    def __init__(self, config):
        super().__init__(config)
        self.env_initializor = EnvInitializor(config)
        self.seed(2022)

    def initialize_env(self, episode_id):
        """Initates the environment
        """
        self.time_step = 0
        self.env_initializor.next_vehicle_id = 1
        self.env_initializor.dummy_stationary_car = self.dummy_stationary_car
        self.vehicles = self.env_initializor.init_env(episode_id)
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.lane_id == 2:
                self.sdv = self.turn_sdv(vehicle)
                self.vehicles[i] = self.sdv


    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def turn_sdv(self, vehicle):
        """Keep the initial state of the vehicle, but let tree search
            guide its actions.
        """
        sdv_vehicle = SDVehicle(id='sdv')
        for attrname, attrvalue in list(vehicle.__dict__.items()):
            if attrname != 'id':
                setattr(sdv_vehicle, attrname, copy.copy(attrvalue))
        return sdv_vehicle

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles other than SDV on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            if vehicle.id != 'sdv':
                vehicle.neighbours = vehicle.my_neighbours(self.vehicles+[self.dummy_stationary_car])
                # IDMMOBIL car
                actions = vehicle.act()
                vehicle.act_long = actions[0]
                if vehicle.act_long < vehicle.min_act_long:
                    vehicle.min_act_long = vehicle.act_long
                # print('act_long ', vehicle.id, ' ', round(vehicle.act_long))
                # actions[0] += self.rng.normal(0, 3)
                joint_action.append(actions)
        return joint_action

    def get_sdv_action(self, decision):
        self.sdv.neighbours = self.sdv.my_neighbours(self.vehicles+[self.dummy_stationary_car])
        actions = self.sdv.act(decision)
        self.sdv.act_long = actions[0]
        if self.sdv.act_long < self.sdv.min_act_long:
            self.sdv.min_act_long = self.sdv.act_long
        # if self.sdv.neighbours['att']:
        #     print('sdv att ', self.sdv.neighbours['att'].id)
        #     print('sdv targ ', self.sdv.target_lane)

        return actions

    def step(self, decision=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        # print(self.time_step, ' ########################### STEP ######')
        joint_action = self.get_joint_action()
        sdv_action = self.get_sdv_action(decision)

        for vehicle, actions in zip(self.vehicles, joint_action):
            if vehicle.id != 'sdv':
                vehicle.step(actions)
                vehicle.time_lapse += 1

        self.sdv.step(sdv_action)
        self.sdv.time_lapse += 1
        self.time_step += 1

    def planner_observe(self):
        """Observation used by the planner
        """
        # delta_x_to_merge = self.ramp_exit_start-self.glob_x
        # obs = {
        #        'gloab_y':None,
        #        'delta_x_to_merge' :None,
        #        }
        return self.sdv.glob_y
        # return self.glob_y + np.random.normal()
        # return delta_x_to_merge

    def is_terminal(self):
        """ Set conditions for instance at which the episode is over
        Note: Collisions are not possible, since agent does not take actions that
            result in collisions

        Episode is complete if: agent successfully performs a merge
        """
        if self.sdv.is_merge_complete():
            if self.sdv.neighbours['rl']:
                self.sdv.neighbours['rl'].neighbours['f'] = self
                self.sdv.neighbours['rl'].neighbours['m'] = None
            self.sdv.lane_decision = 'keep_lane'
            self.sdv.glob_y = 1.5*self.sdv.lane_width
            return True
        else:
            return False
        # return False


    def get_reward(self):
        # if decision == 1:
        #     return 0
        # else:
        #     return 1



        total_reward = 0
        if self.sdv.is_merge_complete():
            total_reward += 0.2

        for vehicle in self.vehicles:
            # print(vehicle.id, ' ', round(vehicle.min_act_long))
            if vehicle.min_act_long < -6:
                total_reward -= 1
            # sys.exit()
        return total_reward
