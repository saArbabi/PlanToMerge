from envs.merge import EnvMerge
from vehicles.sdv_vehicle import SDVehicle
import numpy as np
import copy
from envs.env_initializor_test import EnvInitializor
# from envs.env_initializor import EnvInitializor
import sys

class EnvAutoMerge(EnvMerge):
    def __init__(self):
        super().__init__()
        self.seed(2022)

    def env_reward_reset(self):
        self.is_large_deceleration = False

    def initialize_env(self, episode_id):
        """Initiates the environment
        """
        self.time_step = 0
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
            act_long = actions[0]
            if act_long < -5:
                self.is_large_deceleration = True

            if self.time_step > 0:
                vehicle.act_long_p = vehicle.act_long_c
            vehicle.act_long_c = act_long
        return joint_action

    def get_sdv_action(self, decision):
        self.sdv.neighbours = self.sdv.my_neighbours(self.vehicles+[self.dummy_stationary_car])
        actions = self.sdv.act(decision)
        act_long = actions[0]
        if self.time_step > 0:
            self.sdv.act_long_p = self.sdv.act_long_c
        self.sdv.act_long_c = act_long

        return actions

    def track_history(self, vehicle, actions):
        if vehicle.id == 2:
            obs = vehicle.neur_observe()
            vehicle.update_obs_history(obs[0])

    def step(self, decision=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        # print(self.time_step, ' ########################### STEP ######')
        joint_action = self.get_joint_action()
        sdv_action = self.get_sdv_action(decision)

        for vehicle, actions in zip(self.vehicles, joint_action):
            self.track_history(vehicle, actions)
            vehicle.step(actions)

        self.track_history(self.sdv, sdv_action)
        self.sdv.step(sdv_action)
        self.time_step += 1

    def planner_observe(self):
        """Observation used by the planner
        """
        # delta_x_to_merge = self.ramp_exit_start-self.glob_x
        # obs = {
        #        'gloab_y':None,
        #        'delta_x_to_merge' :None,
        #        }
        # obs = 0
        # for vehicle in self.vehicles:
        #     if vehicle.max_brake < -5:
        #         obs += vehicle.glob_x

        # return obs + np.random.normal()
        # return obs + np.random.normal()
        # return np.random.choice(range(20))
        return self.rng.randint(-20, 20)
        # return np.random.choice(range(20))
        # return 1
        # return self.sdv.glob_y + self.rng.random()
        # return delta_x_to_merge

    def is_terminal(self):
        """ Set conditions for instance at which the episode is over
        Note: Collisions are not possible, since agent does not take actions that
            result in collisions

        Episode is complete if:
        (1) agent successfully performs a merge
        """
        # return False
        if self.sdv.is_merge_complete():
            return True

    def get_reward(self):
        """
        Reward is set to encourage the following behaviours:
        1) perform merge successfully (TTM)
        2) avoid reckless decisions
        """
        total_reward = 0
        if self.sdv.is_merge_complete():
            total_reward += 0.2
        #
        if self.is_large_deceleration:
            total_reward -= 0.2

        return total_reward

    def all_cars(self):
        """
        returns the list of all the cars on the road
        """
        return self.vehicles + [self.sdv]
