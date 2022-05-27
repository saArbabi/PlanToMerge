from envs.merge import EnvMerge
from vehicles.sdv_vehicle import SDVehicle
import numpy as np
import copy
# from envs.env_initializor_test import EnvInitializor
from envs.env_initializor import EnvInitializor
import sys

class EnvAutoMerge(EnvMerge):
    def __init__(self):
        super().__init__()

    def initialize_env(self, episode_id):
        """Initiates the environment
        """
        self.time_step = 0
        self.env_reward_reset()
        env_initializor = EnvInitializor(self.config)
        env_initializor.next_vehicle_id = 1
        env_initializor.dummy_stationary_car = self.dummy_stationary_car
        self.vehicles = env_initializor.init_env(episode_id)
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.lane_id == 2:
                self.sdv = self.turn_sdv(vehicle)
                del self.vehicles[i]

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def env_reward_reset(self):
        """This is reset with every planner timestep
        """
        self.got_bad_action = False

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
            vehicle.neighbours = vehicle.my_neighbours(self.all_cars() + \
                                                       [self.dummy_stationary_car])
            actions = vehicle.act()
            joint_action.append(actions)
        return joint_action

    def get_sdv_action(self):
        self.sdv.neighbours = self.sdv.my_neighbours(self.vehicles+[self.dummy_stationary_car])
        actions = self.sdv.act()
        return actions

    def is_bad_action(self, vehicle, actions):
        return not self.got_bad_action and vehicle.neighbours['m'] and \
                vehicle.neighbours['m'].id == 'sdv' and \
                self.sdv.lane_decision != 'keep_lane' and actions[0] < -5

    def log_actions(self, vehicle, actions):
        act_long = actions[0]
        if vehicle.act_long_c:
            vehicle.act_long_p = vehicle.act_long_c
        vehicle.act_long_c = act_long

    def track_history(self, vehicle):
        """Use history for prediction
        """
        if vehicle.id != 1:
            obs = vehicle.neur_observe()
            vehicle.update_obs_history(obs[0])

    def step(self):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        joint_action = self.get_joint_action()
        sdv_action = self.get_sdv_action()
        self.log_actions(self.sdv, sdv_action)
        self.sdv.step(sdv_action)

        for vehicle, actions in zip(self.vehicles, joint_action):
            self.log_actions(vehicle, actions)
            self.track_history(vehicle)
            vehicle.step(actions)
            if self.is_bad_action(vehicle, actions):
                self.got_bad_action = True

        self.time_step += 1

    def all_cars(self):
        """
        returns the list of all the cars on the road
        """
        return self.vehicles + [self.sdv]

    def get_reward(self, decision):
        """
        Reward is set to encourage the following behaviours:
        1) perform merge successfully
        2) avoid reckless decisions
        """
        total_reward = 0
        if not self.sdv.abort_been_chosen and decision == 5:
            self.sdv.abort_been_chosen = True
            total_reward -= 2

        if self.sdv.is_merge_complete():
            if self.sdv.abort_been_chosen:
                total_reward += 1
            else:
                total_reward += 3


        if self.got_bad_action:
            total_reward -= 5

        return total_reward
