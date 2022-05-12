"""
This is the environment object used by the agent for planning.
While in the simulator IDM determines vehicle action bassed on traffic
state, here NIDM is used to output vehicle actions.
"""
from envs.auto_merge import EnvAutoMerge
import copy

class ImaginedEnv(EnvAutoMerge):
    def __init__(self):
        super().__init__()

    def copy_attrs(self, state):
        for attrname in ['vehicles', 'sdv', 'time_step']:
            attrvalue = getattr(state, attrname)
            setattr(self, attrname, copy.deepcopy(attrvalue))

    def uniform_prior(self):
        """
        Sets sdv's prior belief about other drivers
        """
        for vehicle in self.vehicles:
            print('before ', vehicle.driver_params['aggressiveness'])
            vehicle.driver_params['aggressiveness'] = self.rng.uniform(0.1, 0.9)
            vehicle.set_driver_params()
            print('after ', vehicle.driver_params['aggressiveness'])

    def step(self, joint_action, decision=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        # print(self.time_step, ' ########################### STEP ######')
        # joint_action = self.get_joint_action()
        sdv_action = self.get_sdv_action(decision)

        for vehicle, actions in zip(self.vehicles, joint_action):
            self.track_history(vehicle, actions)
            vehicle.step(actions)

        self.track_history(self.sdv, sdv_action)
        self.sdv.step(sdv_action)
        self.time_step += 1
