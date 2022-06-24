from importlib import reload
from vehicles.vehicle import Vehicle
import json

class IDMMOBILVehicleMerge(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)

        self.act_long_c = 0 # current action
        self.act_long_p = 0 # last action
        self.merge_lane_start = config['merge_lane_start']
        self.ramp_exit_start = config['ramp_exit_start']
        self.beta_precision = 15
        self.lane_id = lane_id
        self.lane_width = 3.75
        self.lanes_n = 2
        self.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.neighbours = {veh_name: None for veh_name in\
                            ['f', 'fl', 'rl', 'r', 'rr', 'fr', 'm', 'att']}
        self.perception_range = 500 #
        self.lateral_actions = {'move_left':0.75,
                                'move_right':-0.75,
                                'keep_lane':0}
        self.parameter_range = {'most_aggressive': {
                                        'desired_v':25, # m/s
                                        'desired_tgap': 0.5, # s
                                        'min_jamx':1, # m
                                        'max_act':4, # m/s^2
                                        'min_act':4, # m/s^2
                                        'politeness':0.,
                                        'safe_braking':-5,
                                        'act_threshold':0
                                        },
                         'least_aggressvie': {
                                        'desired_v':15, # m/s
                                        'desired_tgap':2, # s
                                        'min_jamx':5, # m
                                        'max_act':2, # m/s^2
                                        'min_act':2, # m/s^2
                                        'politeness':1,
                                        'safe_braking':-3,
                                        'act_threshold':0.2
                                         }}
        self.driver_params = {}
        self.driver_params['aggressiveness'] = aggressiveness  # in range [0, 1]

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 1000]
        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        assert delta_x > 0, 'leader {l_id} is behind follower {f_id}'.format(\
                                        l_id=leader.id, f_id=follower.id)
        return [delta_v, delta_x]

    def idm_action(self, follower, leader):
        """
        Note: Follower is not always the ego, it can be the ego's neighbour(use:MOBIL)
        """
        if not follower:
            return 0
        delta_v, delta_x = self.observe(follower, leader)
        desired_gap = follower.driver_params['min_jamx'] + \
                        max(0,
                        follower.driver_params['desired_tgap']*\
                        follower.speed+(follower.speed*delta_v)/ \
                        (2*(follower.driver_params['max_act']*\
                        follower.driver_params['min_act'])**0.5))

        act_long = follower.driver_params['max_act']*\
                    (1-(follower.speed/follower.driver_params['desired_v'])**4-\
                                                        (desired_gap/(delta_x))**2)

        return max(-7, min(act_long, 5))

    def my_neighbours(self, vehicles):
        """
        Ego can be in 3 states:
        - decided to change lane - indicaiton
        - decided to change lane - in progress
        - decided to keep lane

        Neighbours can be in 3 states:
        - decided to change lane - indicaiton
        - decided to change lane - in progress
        - decided to keep lane

        Who is ego's neighbour depends on ego's and a given neighbour's state.
        """
        neighbours = {}
        delta_xs_f, delta_xs_fl, delta_xs_rl, delta_xs_r, \
        delta_xs_fr, delta_xs_att = ([self.perception_range] for i in range(6))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
        candidate_fr, candidate_m, candidate_att = (None for i in range(7))

        left_lane_id = self.lane_id - 1

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = abs(vehicle.glob_x-self.glob_x)
                if delta_x < self.perception_range:
                    if self.is_it_merger(vehicle):
                        candidate_m = vehicle

                    if vehicle.glob_x > self.glob_x:
                        # front neibouring cars
                        if vehicle.target_lane == self.target_lane and \
                                vehicle.lane_decision == 'keep_lane':

                            if delta_x < min(delta_xs_f):
                                delta_xs_f.append(delta_x)
                                candidate_f = vehicle

                            if delta_x < min(delta_xs_att):
                                delta_xs_att.append(delta_x)
                                candidate_att = vehicle

                        if vehicle.target_lane == left_lane_id:
                            if delta_x < min(delta_xs_fl):
                                delta_xs_fl.append(delta_x)
                                candidate_fl = vehicle
                    else:
                        if vehicle.target_lane == left_lane_id:
                            if delta_x < min(delta_xs_rl):
                                delta_xs_rl.append(delta_x)
                                candidate_rl = vehicle

                        if vehicle.lane_id == self.lane_id == vehicle.target_lane:
                            # same lane
                            if delta_x < min(delta_xs_r):
                                delta_xs_r.append(delta_x)
                                candidate_r = vehicle

        neighbours['f'] = candidate_f
        neighbours['fl'] = candidate_fl
        neighbours['rl'] = candidate_rl
        neighbours['fr'] = candidate_fr
        neighbours['r'] = candidate_r

        if candidate_m and candidate_f and \
                    candidate_m.glob_x > candidate_f.glob_x and \
                    candidate_m.lane_decision != 'keep_lane':
            neighbours['m'] = None
        else:
            neighbours['m'] = candidate_m

        if neighbours['m'] and self.am_i_attending(neighbours['m'], candidate_f):
            neighbours['att'] = neighbours['m']
        else:
            neighbours['att'] = candidate_att
        return neighbours

    def is_it_merger(self, vehicle):
        if vehicle.id == 'dummy':
            return False
        elif self.glob_x > self.ramp_exit_start and vehicle.glob_x < self.glob_x:
            return False
        elif (vehicle.lane_decision != 'keep_lane' and vehicle.glob_x > self.glob_x) \
            or (vehicle.lane_decision == 'keep_lane' and vehicle.lane_id == 2):
            return True
        return False

    def get_ttm(self, m_veh):
        delta_y = self.lane_width*1.5 - m_veh.glob_y# merger lateral distance to main road
        ttm_m_veh = delta_y/self.lateral_actions['move_left'] # steps to merge
        return ttm_m_veh

    def is_cidm_att(self, act_long, m_veh, f_veh):
        ttm_m_veh = self.get_ttm(m_veh)
        gap_to_merge = ttm_m_veh*m_veh.speed + m_veh.glob_x-self.glob_x
        ttm_e_veh = gap_to_merge/self.speed
        # print('ttm_e_veh + polit', self.driver_params['politeness']*ttm_e_veh)
        # print('ttm_e_veh ', ttm_e_veh)
        # print('ttm_m_veh ', ttm_m_veh)
        # print('act_long ', act_long)
        if ttm_m_veh < self.driver_params['politeness']*ttm_e_veh and \
                                act_long > -self.driver_params['min_act']:
            return True

    def am_i_attending(self, m_veh, f_veh):
        """Several scenarios are possible:
        (1) Ego attends because merger has entered its lane
        (2) Ego attends following the cooperative idm
        (3) Ego attends for safety
        """
        if (self.glob_x > m_veh.glob_x) or (f_veh.glob_x < m_veh.glob_x) \
                            or (m_veh.glob_x < self.merge_lane_start) \
                            or m_veh.lane_decision == 'keep_lane':
            return False

        elif self.neighbours['att'] and m_veh.id == self.neighbours['att'].id:
            return True
        elif m_veh.lane_id == self.lane_id:
            # print('lane-based ########')
            return True

        act_long = self.idm_action(self, m_veh)
        if act_long < -6:
            # emergency situation
            # print('collisio- avoidance based ########')
            return True
        elif self.is_cidm_att(act_long, m_veh, f_veh):
            # print('cidm-based ########')
            return True
        else:
            return False

    def act(self):
        act_long, act_lat = self.idm_mobil_act()
        return [act_long, act_lat]

    def is_merge_complete(self):
        return self.glob_y >= 1.5*self.lane_width

    def lateral_action(self):
        if self.lane_decision == 'keep_lane':
            return 0

        if self.glob_x >= self.ramp_exit_start:
            if self.target_lane != 1:
                self.target_lane = 1
            return self.lateral_actions[self.lane_decision]
        else:
            return 0

    def is_merge_possible(self, act_rl_lc):
        if self.lane_id > 1 and self.glob_x > self.merge_lane_start and \
                self.driver_params['safe_braking'] <= act_rl_lc:
            return True

    def mobil_condition(self, actions_gains):
        """To decide if changing lane is worthwhile.
        """
        ego_gain, new_follower_gain, old_follower_gain = actions_gains
        lc_condition = ego_gain+self.driver_params['politeness']*(new_follower_gain+\
                                                                old_follower_gain )
        return lc_condition

    def idm_mobil_act(self):
        act_long = self.idm_action(self, self.neighbours['att'])
        if self.lane_decision != 'keep_lane':
            if self.is_merge_complete():
                if self.neighbours['rl']:
                    self.neighbours['rl'].neighbours['f'] = self
                    self.neighbours['rl'].neighbours['m'] = None
                self.lane_decision = 'keep_lane'
                self.glob_y = 1.5*self.lane_width

        elif self.lane_decision == 'keep_lane' and self.lane_id == 2:
            lc_left_condition = 0
            act_ego_lc_l = self.idm_action(self, self.neighbours['fl'])
            act_rl_lc = self.idm_action(self.neighbours['rl'], self)
            # print('act_rl_lc ', act_rl_lc)
            if self.is_merge_possible(act_rl_lc):
                # consider moving left
                act_r_lc = self.idm_action(self.neighbours['r'], self.neighbours['f'])
                act_r_lk = self.idm_action(self.neighbours['r'], self)
                old_follower_gain = act_r_lc-act_r_lk

                act_rl_lk = self.idm_action(self.neighbours['rl'], self.neighbours['fl'])
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk

                lc_left_condition = self.mobil_condition([ego_gain, \
                                        new_follower_gain, old_follower_gain])

                # print('ego_gain ', ego_gain)
                # print('old_follower_gain ', old_follower_gain)
                # print('new_follower_gain ', new_follower_gain)
                # print('lc_left_condition ', lc_left_condition)

            if lc_left_condition > self.driver_params['act_threshold']:
                self.lane_decision = 'move_left'

        return [act_long, self.lateral_action()]
