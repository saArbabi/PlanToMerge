import json
import pickle
import numpy as np
import tensorflow as tf

class NIDM():
    def __init__(self):
        self.load_nidm()
        self.load_scalers()
        self.create_state_indxs()
        self.samples_n = 1

    def load_nidm(self):
        model_name = 'neural_idm_371'
        epoch_count = '17'
        exp_dir = './src/models/'+model_name
        exp_path = exp_dir+'/model_epo'+epoch_count
        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)

        from models.neural_idm import  NeurIDMModel
        self.model = NeurIDMModel(config)
        self.model.load_weights(exp_path).expect_partial()
        tf.random.set_seed(2022)

    def load_scalers(self):
        data_files_dir = './src/models/'

        with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
            self.env_scaler = pickle.load(handle)

        with open(data_files_dir+'m_scaler.pickle', 'rb') as handle:
            self.m_scaler = pickle.load(handle)

        with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

    def names_to_index(self, col_names):
        if type(col_names) == list:
            return [self.indxs[item] for item in col_names]
        else:
            return self.indxs[col_names]

    def create_state_indxs(self):
        self.indxs = {}
        feature_names = [
                        'e_veh_action_p', 'f_veh_action_p',
                        'e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x',
                        'em_delta_v', 'em_delta_x',
                        'm_veh_action_p', 'm_veh_speed','em_delta_y',
                        'delta_x_to_merge','m_veh_exists', 'm_veh_decision']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1
        col_names = ['e_veh_action_p', 'f_veh_action_p', 'e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']
        self.env_s_indxs = self.names_to_index(col_names)
        col_names = ['m_veh_action_p', 'm_veh_speed', 'em_delta_y', 'delta_x_to_merge']
        self.merger_indxs = self.names_to_index(col_names)

    def driver_params_update(self, vehicles, idm_params):
        idm_params = idm_params.numpy()
        for i, vehicle in enumerate(vehicles):
            vehicle.driver_params['desired_v'] = idm_params[i, 0]
            vehicle.driver_params['desired_tgap'] = idm_params[i, 1]
            vehicle.driver_params['min_jamx'] = idm_params[i, 2]
            vehicle.driver_params['max_act'] = idm_params[i, 3]
            vehicle.driver_params['min_act'] = idm_params[i, 4]
        # print(vehicle.driver_params)

    def scale_state(self, state, state_type):
        if state_type == 'full':
            state[:, :, self.env_s_indxs] = \
                (state[:, :, self.env_s_indxs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5
            # merger context
            state[:, :, self.merger_indxs] = \
                (state[:, :, self.merger_indxs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5
            state[:,:,]
        elif state_type == 'env_state':
            state = \
                (state[:, :, self.env_s_indxs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5

        elif state_type == 'merger_c':
            state = \
                (state[:, :, self.merger_indxs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5

        return np.float32(state)

    def get_neur_att(self, att_context):
        f_att_score, m_att_score = self.model.forward_sim.get_att(att_context)
        f_att_score, m_att_score = f_att_score.numpy()[0][0][0], m_att_score.numpy()[0][0][0]
        # f_att_score = (1 - self.m_veh_exists) + f_att_score*self.m_veh_exists
        # m_att_score = m_att_score*self.m_veh_exists
        return f_att_score, m_att_score

    def action_clip(self, act_long):
        return min(max([-6, act_long]), 6)

    def aggregate_histories(self, vehicles):
        histories = [vehicle.obs_history for vehicle in vehicles]
        return np.array(histories)

    def latent_inference(self, vehicles):
        obs_history = self.scale_state(self.aggregate_histories(vehicles), 'full')
        enc_hs = self.model.h_seq_encoder(obs_history)
        latent_dis_params = self.model.belief_net(enc_hs , dis_type='prior')
        enc_hs = tf.reshape(enc_hs, [len(vehicles), 1, 128])
        hidden_state = [enc_hs, latent_dis_params]
        return hidden_state

    def sample_latent(self, root_state):
        """Take a latent sample and apply NN projections
        """
        z_idm, z_att = self.model.belief_net.sample_z(root_state.hidden_state[-1])
        proj_idm, proj_att = self.apply_projections(z_idm, z_att)
        idm_params = self.model.idm_layer(proj_idm)
        root_state.proj_att = tf.reshape(proj_att, [len(root_state.vehicles), 1, 128])
        return idm_params

    def apply_projections(self, z_idm, z_att):
        proj_idm = self.model.belief_net.z_proj_idm(z_idm)
        proj_att = self.model.belief_net.z_proj_att(z_att)
        return proj_idm, proj_att

    def should_att_pred(self, vehicle):
        if not vehicle.neighbours['m'] or vehicle.neighbours['m'].lane_decision == 'keep_lane':
            return False
        return True

    def get_att_context(self, root_state, obs_t0, index):
        obs_t0 = np.array(obs_t0)
        booleans = obs_t0[:, :, -2:] # ['m_veh_exists', 'm_veh_decision']
        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        att_context = tf.concat([root_state.proj_att[index:index+1, :, :],
                                 root_state.hidden_state[0][index:index+1, :, :],
                                 env_state, merger_c, booleans], axis=-1)
        return att_context

    def predict_joint_action(self, root_state, vehicle, index):
        obs_t0 = [[vehicle.obs_history[-1]]]
        att_context = self.get_att_context(root_state, obs_t0, index)
        f_att_score, m_att_score = self.get_neur_att(att_context)
        ef_act = self.action_clip(vehicle.idm_action(vehicle, vehicle.neighbours['f']))
        if vehicle.neighbours['m'].glob_x > vehicle.glob_x:
            em_act = vehicle.idm_action(vehicle, vehicle.neighbours['m'])
            if em_act < -20:
                # not a feasible action
                em_act = 0
                m_att_score = 0
            else:
                em_act = self.action_clip(em_act)
        else:
            # no merger to attend to
            em_act = 0
            m_att_score = 0

        act_long = f_att_score*ef_act + m_att_score*em_act
        # print('oooops  ',
        #         'm_att_score ', round(m_att_score, 2), '  ',
        #         'em_act ', round(em_act, 2), '  ',
        #         'act_long ', round(act_long, 2), '  ',
        #         'glob_y_m ', round(vehicle.neighbours['m'].glob_y, 2), '  ',
        #         'glob_x_m ', round(vehicle.neighbours['m'].glob_x-vehicle.glob_x, 2), '  ',
        #         'glob_x_f ', round(vehicle.neighbours['f'].glob_x-vehicle.glob_x, 2))
        return [act_long, 0]
