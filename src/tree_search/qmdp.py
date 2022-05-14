from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.abstract import AbstractPlanner, Node
from tree_search.factory import safe_deepcopy_env
import time
import hashlib
import numpy as np
import sys
from tree_search.imagined_env import ImaginedEnv

class QMDP(MCTSDPW):
    def __init__(self):
        super(QMDP, self).__init__()
        self.update_counts = 0
        self._enough_history = False
        self.decision_counts = False
        self.nidm = self.load_nidm()
        self.img_state = ImaginedEnv()

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = BeliefNode(parent=None, config=self.config)

    def enough_history(self, state):
        """
        checks to see if enough observations have been tracked for the model.
        """
        if not self._enough_history:
            for vehicle in state.vehicles:
                if vehicle.id == 2:
                    if not np.isnan(vehicle.obs_history).any():
                        self._enough_history = True
        return self._enough_history

    def predict_vehicle_actions(self, state):
        """
        Returns the joint action of all vehicles other than SDV on the road
        """
        joint_action = []
        for vehicle in state.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(state.all_cars() + \
                                                       [state.dummy_stationary_car])

            if vehicle.id != 1 and vehicle.enc_h != None:
                actions = self.nidm.estimate_vehicle_action(vehicle)
            else:
                actions = vehicle.act()

            joint_action.append(actions)
        return joint_action

    def step(self, state, decision):
        state.env_reward_reset()
        for i in range(self.steps_per_decision):
            joint_action = self.predict_vehicle_actions(state)
            state.step(joint_action, decision)
        observation = state.planner_observe()
        reward = state.get_reward()
        terminal = state.is_terminal()
        return observation, reward, terminal

    def load_nidm(self):
        from tree_search.nidm import NIDM
        return NIDM()

    def update_belief(self, belief_node, img_state):
        """
        Passes the sequence of past vehicle observations (vehicle standpoint)
        and actions into an LSTM encoder. Then the encoded state is mapped to
        the latent belief.
        """
        if not belief_node.img_state:
            belief_node.img_state = img_state

        if self.enough_history(img_state) and \
                            not belief_node.belief_is_updated:
            belief_node.belief_is_updated = True
            for vehicle in img_state.vehicles:
                if vehicle.id != 1:
                    self.nidm.latent_inference(vehicle)

    def sample_belief(self, belief_node):
        """
        Returns a sample from the belief state.
        Procedure:
        For each vehicle:
            (1) sample from latent dis
            (2) using the sampled latent, obtain relevant projections
            (3) assign the sampled driver params to each vehicle
        """
        sampled_state = safe_deepcopy_env(belief_node.img_state)
        if belief_node.belief_is_updated:
            # you need enough history for this
            for vehicle in sampled_state.vehicles:
                if vehicle.id != 1:
                    z_idm, z_att = self.nidm.sample_latent(vehicle)
                    proj_idm = self.nidm.apply_projections(vehicle, z_idm, z_att)
                    idm_params = self.nidm.model.idm_layer(proj_idm)
                    self.nidm.driver_params_update(vehicle, idm_params)
        else:
            sampled_state.uniform_prior()
        return sampled_state

    def imagine_state(self, state):
        """
        Returns an "imagined" environment state, with uniform prior belief.
        """
        self.img_state.copy_attrs(state)
        self.img_state.uniform_prior()
        return self.img_state

    def run(self, belief_node):
        """
        Note: QMDP only holds belief in the first tree node
        """
        total_reward = 0
        depth = 0
        terminal = False
        state = self.sample_belief(belief_node)
        state.seed(self.rng.randint(1e5))

        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}
        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)
            observation, reward, terminal = self.step(state, decision)
            belief_node = chance_node.get_child(observation,
                                                self.rng)

            total_reward += self.config["gamma"] ** depth * reward
            depth += 1
            self.log_visited_sdv_state(state, tree_states, 'selection')
            self.extract_belief_info(state, depth)

        if not terminal:
            tree_states, total_reward = self.evaluate(state,
                                         tree_states,
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)
        self.extract_tree_info(tree_states)

    def plan(self, state):
        self.reset()
        img_state = self.imagine_state(state)
        belief_node = self.root
        self.update_belief(belief_node, img_state)
        for plan_itr in range(self.config['budget']):
            # input('This is plan iteration '+ str(plan_itr))
            # t_0 = time.time()
            # t_1 = time.time()
            # print('copy time: ', t_1 - t_0)

            self.run(belief_node)
            # print(self.update_counts)

class BeliefNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)
        self.belief_is_updated = False
        self.img_state = None

    def expand(self, available_decisions, rng):
        decision = rng.choice(list(self.unexplored_decisions(available_decisions)))
        self.children[decision] = SubChanceNode(self, self.config)
        return self.children[decision], decision

    def get_child(self, available_decisions, rng):
        if len(self.children) == len(available_decisions) \
                or self.k_decision*self.count**self.alpha_decision < len(self.children):
            # select one of previously expanded decisions
            return self.selection_strategy(rng)
        else:
            # insert a new aciton
            return self.expand(available_decisions, rng)

class SubChanceNode(ChanceNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def get_child(self, observation, rng):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = rng.choice(list(self.children))
                return self.children[obs_id]
            else:
                # Add observation to the children set
                self.expand(obs_id)

        return self.children[obs_id]

    def expand(self, obs_id):
        belief_node = BeliefNode(self, self.config)
        self.children[obs_id] = belief_node
