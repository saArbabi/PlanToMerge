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

    def env_reward_reset(self):
        """This is reset with every planner timestep
        """
        self.bad_action_in_env = False

    def copy_attrs(self, state):
        for attrname in ['vehicles', 'sdv', 'time_step']:
            attrvalue = getattr(state, attrname)
            setattr(self, attrname, copy.deepcopy(attrvalue))

    def uniform_prior(self):
        """
        Sets sdv's prior belief about other drivers
        """
        for vehicle in self.vehicles:
            vehicle.driver_params['aggressiveness'] = self.rng.uniform(0.01, 0.99)
            vehicle.set_driver_params()

    def is_bad_action(self, vehicle, actions):
        return vehicle.id != 1 and actions[0] < -5

    def step(self, joint_action, decision=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        sdv_action = self.get_sdv_action(decision)

        for vehicle, actions in zip(self.vehicles, joint_action):
            self.log_actions(vehicle, actions)
            self.track_history(vehicle)
            vehicle.step(actions)
            if self.is_bad_action(vehicle, actions):
                self.bad_action_in_env = True

        self.log_actions(self.sdv, sdv_action)
        self.sdv.step(sdv_action)
        self.time_step += 1

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
        1) perform merge successfully  
        2) avoid reckless decisions
        """
        total_reward = 0
        if self.sdv.is_merge_complete():
            if not self.sdv.abort_been_chosen:
                total_reward += 0.3
            else:
                total_reward += 0.1

        if self.bad_action_in_env:
            total_reward -= 0.6
        return total_reward

    def planner_observe(self):
        x_road = 0
        for vehicle in self.vehicles:
            x_road += vehicle.glob_x
        return x_road
        # return self.rng.randint(-100, 100)
        # return self.sdv.glob_x
        # return self
