"""
This is the environment object used by the agent for planning.
While in the simulator IDM determines vehicle action bassed on traffic
state, here NIDM is used to output vehicle actions.
"""
from envs.auto_merge import EnvAutoMerge

class ImaginedEnv(EnvAutoMerge):
    def __init__(self):
        super().__init__()

    def copy_attrs(self, state):
        for attrname in ['vehicles', 'sdv', 'time_step']:
            attrvalue = getattr(state, attrname)
            setattr(self, attrname, attrvalue)

    # def aggregate_obs(self):
    #     """
    #
    #     """
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
