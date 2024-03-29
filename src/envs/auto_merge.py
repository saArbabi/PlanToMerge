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
        self.name = 'real'

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
        self.give_way_chosen = False
        self.got_bad_action = False
        self.got_bad_state = False

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

    def is_bad_action(self):
        cond = not self.got_bad_action and \
                self.sdv.lane_decision != 'keep_lane' and \
                self.sdv.neighbours['rl']
        if cond and self.sdv.neighbours['rl'].act_long_c < -4:
            return True

    def is_bad_state(self):
        cond = not self.got_bad_state and \
                self.sdv.lane_decision != 'keep_lane' and \
                self.sdv.neighbours['rl']
        if cond:
            vehicle = self.sdv.neighbours['rl']
            ttc = (self.sdv.glob_x - vehicle.glob_x)/(vehicle.speed - self.sdv.speed)
            tiv = (self.sdv.glob_x - vehicle.glob_x)/(vehicle.speed)
            # if ttc > 1.7 and tiv > 0.3:
            # print('""""""""""""""""""""""""""""')
            # print('ttc ', ttc)
            # print('tiv ', tiv)
            if (ttc > 3.3 or ttc < 0) and tiv > 1.3:
                return False
            else:
                    return True

    def log_actions(self, vehicle, actions):
        act_long = actions[0]
        if vehicle.act_long_c:
            vehicle.act_long_p = vehicle.act_long_c
        vehicle.act_long_c = act_long

    def track_history(self, vehicle):
        """Use history for prediction
        """
        obs = vehicle.neur_observe()
        vehicle.update_obs_history(obs)

    def step(self):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        joint_action = self.get_joint_action()
        sdv_action = self.get_sdv_action()
        self.sdv.step(sdv_action)
        self.log_actions(self.sdv, sdv_action)

        if self.is_bad_action():
            self.got_bad_action = True
        if self.is_bad_state():
            self.got_bad_state = True

        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)
            self.log_actions(vehicle, actions)
            self.track_history(vehicle)


        self.time_step += 1

    def all_cars(self):
        """
        returns the list of all the cars on the road
        """
        return self.vehicles + [self.sdv]

    def get_reward(self, decision=None):
        """
        Reward is set to encourage the following behaviours:
        1) perform merge successfully
        2) avoid reckless decisions
        """
        if not decision:
            return 0

        total_reward = -0.3
        # total_reward = 0
        # if decision == 5:
        #     if self.sdv.prev_decision != 5 or  \
        #         (self.sdv.neighbours['rl'] and \
        #             self.sdv.prev_rl_veh.id != self.sdv.neighbours['rl'].id):
        #         total_reward -= 1

        if self.sdv.is_merge_complete():
            total_reward += 5

        if self.got_bad_state:
            # print('bad bad')
            total_reward += -10
        elif self.got_bad_action:
            total_reward += -5

        return total_reward
