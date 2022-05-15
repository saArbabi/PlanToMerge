from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
import pickle
import numpy as np


class TrackedVehicle(IDMMOBILVehicleMerge):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed, aggressiveness)
        self.samples_n = 1
        self.history_len = 30 # steps
        self.state_dim = 13
        self.obs_history = np.empty([self.samples_n, self.history_len, self.state_dim])
        self.obs_history[:] = np.nan
        self.enc_h = None
        with open('./src/models/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t[0, 0, :]

    def neur_observe(self):
        m_veh = self.neighbours['m']
        f_veh = self.neighbours['f']
        if not m_veh:
            m_veh_exists = 0
            m_veh_action = self.dummy_value_set['m_veh_action_p']
            m_veh_speed = self.dummy_value_set['m_veh_speed']
            em_delta_x = self.dummy_value_set['em_delta_x']
            em_delta_v = self.dummy_value_set['em_delta_v']
            em_delta_y = self.dummy_value_set['em_delta_y']
            delta_x_to_merge = self.dummy_value_set['delta_x_to_merge']

        else:
            m_veh_exists = 1
            m_veh_action = m_veh.act_long_p
            m_veh_speed = m_veh.speed
            em_delta_x = m_veh.glob_x-self.glob_x
            em_delta_y = abs(m_veh.glob_y-self.glob_y)
            em_delta_v = self.speed-m_veh_speed
            delta_x_to_merge = m_veh.ramp_exit_start-m_veh.glob_x

        if not f_veh:
            f_veh_exists = 0
            f_veh_action = self.dummy_value_set['f_veh_action_p']
            f_veh_speed = self.dummy_value_set['f_veh_speed']
            el_delta_x = self.dummy_value_set['el_delta_x']
            el_delta_v = self.dummy_value_set['el_delta_v']
        else:
            f_veh_exists = 1
            f_veh_action = f_veh.act_long_p
            f_veh_speed = f_veh.speed
            el_delta_x = f_veh.glob_x-self.glob_x
            el_delta_v = self.speed-f_veh_speed

        obs_t0 = [self.act_long_p, f_veh_action, self.speed, f_veh_speed]

        obs_t0.extend([el_delta_v,
                             el_delta_x])

        obs_t0.extend([em_delta_v,
                             em_delta_x,
                             m_veh_action,
                             m_veh_speed,
                             em_delta_y,
                             delta_x_to_merge])
        obs_t0.append(m_veh_exists)
        self.m_veh_exists = m_veh_exists
        return [np.array([[obs_t0]]), [[[float(m_veh_exists)]]]]
