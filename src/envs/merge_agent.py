from envs.merge import EnvMerge
from vehicles.sdv_vehicle import SDVehicle

import copy

class EnvMergeAgent(EnvMerge):
    def __init__(self, config):
        super().__init__(config)

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
                joint_action.append(actions)
                if self.time_step > 0:
                    vehicle.act_long_p = vehicle.act_long_c
                vehicle.act_long_c = actions[0]
        return joint_action

    def get_sdv_action(self, decision):
        actions = self.sdv.act(decision)
        return actions

    def step(self, decision=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        joint_action = self.get_joint_action()
        obs = self.sdv.observe()
        sdv_action = self.get_sdv_action(decision)

        for vehicle, actions in zip(self.vehicles, joint_action):
            if vehicle.id != 'sdv':
                vehicle.step(actions)
                vehicle.time_lapse += 1

        self.sdv.step(sdv_action)
        self.sdv.time_lapse += 1

        self.time_step += 1
        return obs, self.get_reward(obs), False

    def get_reward(self, obs):
        return -obs
