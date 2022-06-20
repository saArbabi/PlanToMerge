"""
This is the environment object used by the agent for planning.
While in the simulator IDM determines vehicle action bassed on traffic
state, here NIDM is used to output vehicle actions.
"""
from envs.auto_merge import EnvAutoMerge
import copy

class ImaginedEnv(EnvAutoMerge):
    def __init__(self, state):
        super().__init__()
        for attrname in ['vehicles', 'sdv', 'time_step']:
            attrvalue = getattr(state, attrname)
            setattr(self, attrname, copy.deepcopy(attrvalue))

    def uniform_prior(self):
        """
        Sets sdv's prior belief about other drivers
        """
        for vehicle in self.vehicles:
            vehicle.driver_params['aggressiveness'] = self.rng.uniform(0.01, 0.99)
            vehicle.set_driver_params(self.rng)

    def give_des(self, vehicle):
        des = 0
        for i in vehicle.driver_params.values():
            des += i
        return des


    def step(self, joint_action):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        sdv_action = self.get_sdv_action()

        for vehicle, actions in zip(self.vehicles, joint_action):
            self.log_actions(vehicle, actions)
            self.track_history(vehicle)
            if self.is_bad_action(vehicle, actions):
                self.got_bad_action = True
                # print('########################')
                # print('id ', vehicle.id)
                # print('act ', actions[0])
                # print('attend_veh ', vehicle.neighbours['att'].id)
                # print('time_step ', self.time_step)
            if vehicle.id == 3 and 40 > self.time_step > 20:
                if self.got_bad_action:
                    print('*',
                          self.time_step, ' ',
                          ' ego att: ', vehicle.neighbours['att'].id,
                          ' ego act: ', vehicle.act_long_c, ' ',
                          ' sdv act: ', self.sdv.act_long_c, ' ',
                          ' sdv dec: ', self.sdv.decision)

                else:
                    print(
                          self.time_step, ' ',
                          ' ego att: ', vehicle.neighbours['att'].id,
                          ' ego act: ', vehicle.act_long_c, ' ',
                          ' sdv act: ', self.sdv.act_long_c, ' ',
                          ' sdv dec: ', self.sdv.decision)
            vehicle.step(actions)

                # if vehicle.am_i_attending(vehicle.neighbours['m'], \
                #                        vehicle.neighbours['f']):
                #     print('yess')
                #     print(vehicle.idm_action(vehicle, vehicle.neighbours['m']))
                # else:
                #     print('No')

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
        if self.sdv.is_merge_complete():
            return True

    def planner_observe(self):
        return self.vehicles[0].glob_x
