"""
This is the environment object used by the agent for planning.
While in the simulator IDM determines vehicle action bassed on traffic
state, here NIDM is used to output vehicle actions.
"""
from envs.auto_merge import EnvAutoMerge
import copy

class ImaginedEnv(EnvAutoMerge):
    def __init__(self, state):
        self.hidden_state = [] # this is to be estimated via NIDM
        for attrname in ['vehicles', 'sdv', 'time_step', 'dummy_stationary_car']:
            attrvalue = getattr(state, attrname)
            setattr(self, attrname, copy.deepcopy(attrvalue))

    def uniform_prior(self, vehicles, seed):
        """
        Sets sdv's prior belief about other drivers
        """
        self.seed(seed)
        for vehicle in vehicles:
            vehicle.driver_params['aggressiveness'] = self.rng.uniform(0.01, 0.99)
            vehicle.set_driver_params(self.rng)

    def mean_prior(self, vehicles, seed):
        self.seed(seed)
        for vehicle in vehicles:
            vehicle.driver_params['aggressiveness'] = 0.5
            vehicle.set_driver_params(self.rng)

    def step(self, joint_action):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        sdv_action = self.get_sdv_action()
        self.sdv.step(sdv_action)
        self.log_actions(self.sdv, sdv_action)

        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)
            self.log_actions(vehicle, actions)
            self.track_history(vehicle)
            if self.is_bad_action(vehicle, actions):
                self.bad_action = actions[0]

            if self.is_bad_state(vehicle):
                self.got_bad_state = True

        self.time_step += 1

    def is_terminal(self, decision):
        """ Set conditions for instance at which the episode is over
        Note: Collisions are not possible, since agent does not take actions that
            result in collisions
        """
        if self.sdv.is_merge_complete() or self.got_bad_state or decision == 5:
            return True

    def planner_observe(self):
        return self.vehicles[0].glob_x
