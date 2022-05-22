from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.abstract import AbstractPlanner, Node
from tree_search.factory import safe_deepcopy_env
from tree_search.imagined_env import ImaginedEnv
import time
import hashlib
import numpy as np
import sys

class QMDP(MCTSDPW):
    def __init__(self):
        super(QMDP, self).__init__()
        self.update_counts = 0
        self._enough_history = False
        self.nidm = self.load_nidm()

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

    def load_nidm(self):
        from tree_search.nidm import NIDM
        return NIDM()

    def update_belief(self, belief_node):
        """
        Passes the sequence of past vehicle observations (vehicle standpoint)
        and actions into an LSTM encoder. Then the encoded state is mapped to
        the latent belief.
        """
        if self.enough_history(belief_node.state) and \
                            not belief_node.belief_is_updated:
            belief_node.belief_is_updated = True
            for vehicle in belief_node.state.vehicles:
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
        img_state = ImaginedEnv(belief_node.state)
        if belief_node.belief_is_updated:
            # you need enough history for this
            for vehicle in img_state.vehicles:
                if vehicle.id != 1:
                    z_idm, z_att = self.nidm.sample_latent(vehicle)
                    proj_idm = self.nidm.apply_projections(vehicle, z_idm, z_att)
                    idm_params = self.nidm.model.idm_layer(proj_idm)
                    self.nidm.driver_params_update(vehicle, idm_params)
        else:
            img_state.seed(self.rng.randint(1e5))
            img_state.uniform_prior()
        return img_state

    def run(self, belief_node):
        """
        Note: QMDP only holds belief in the first tree node
        """
        total_reward = 0
        depth = 0
        terminal = False
        state = self.sample_belief(belief_node)

        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision)
            child_type, belief_node = chance_node.get_child(
                                                state,
                                                observation,
                                                self.rng)

            state = belief_node.fetch_state()
            total_reward += self.config["gamma"] ** depth * reward
            depth += 1

        if not terminal:
            total_reward = self.evaluate(state,
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)

    def plan(self, state):
        self.reset()
        belief_node = self.root
        belief_node.state = ImaginedEnv(state)
        self.update_belief(belief_node)
        for plan_itr in range(self.config['budget']):
            self.run(belief_node)

class BeliefNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)
        self.belief_is_updated = False
        self.state = None

    def expand(self, available_decisions, rng):
        decision = rng.choice(list(self.unexplored_decisions(available_decisions)))
        self.children[decision] = SubChanceNode(self, self.config)
        return self.children[decision], decision

class SubChanceNode(ChanceNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def expand(self, state, obs_id):
        self.children[obs_id] = BeliefNode(self, self.config)
        self.children[obs_id].state = state
