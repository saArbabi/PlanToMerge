from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.imagined_env import ImaginedEnv
import hashlib
import numpy as np
import sys

class QMDP(MCTSDPW):
    def __init__(self, config=None):
        self.nidm = self.load_nidm()
        super(QMDP, self).__init__(config)

    def initialize_planner(self):
        self._enough_history = False
        self.steps_till_next_decision = 0
        self.seed(2022)
        self.nidm.seed(2022)

    def reset(self):
        self.root = BeliefNode(parent=None, config=self.config)

    def enough_history(self, state):
        """
        checks to see if enough observations have been tracked for the model.
        """
        if not self._enough_history:
            if len(state.vehicles[0].obs_history) == 10:
                self._enough_history = True
        return self._enough_history

    def predict_joint_action(self, state):
        """
        Returns the joint action of all vehicles other than SDV on the road
        """
        joint_action = []
        for index, vehicle in enumerate(state.vehicles):
            vehicle.neighbours = vehicle.my_neighbours(state.all_cars() + \
                                                       [state.dummy_stationary_car])
            if self.root.state.hidden_state and self.nidm.should_att_pred(vehicle):
                actions = self.nidm.predict_joint_action(self.root.state, vehicle, index)
            else:
                actions = vehicle.act()

            joint_action.append(actions)

        return joint_action

    def load_nidm(self):
        from tree_search.nidm import NIDM
        return NIDM()

    def update_belief(self, belief_node):
        # return
        if self.enough_history(belief_node.state) and \
                            not belief_node.state.hidden_state:
            state = belief_node.state
            state.hidden_state = self.nidm.latent_inference(state.vehicles)

    def sample_belief(self, belief_node):
        """
        Returns a sample from the belief state.
        """
        img_state = ImaginedEnv(belief_node.state)
        if belief_node.state.hidden_state:
            idm_params = self.nidm.sample_latent(belief_node.state)
            self.nidm.driver_params_update(img_state.vehicles, idm_params)
        else:
            img_state.uniform_prior(img_state.vehicles, self.rng.randint(1e5))

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

            observation, reward, terminal = self.step(state, decision, 'search')
            belief_node = chance_node.get_child(
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
        self.last_decision = state.sdv.decision
        self.current_time_step = state.time_step
        available_decisions = self.get_available_decisions(state)
        if len(available_decisions) > 1:
            self.reset()
            belief_node = self.root
            belief_node.state = ImaginedEnv(state)
            self.update_belief(belief_node)
            for plan_itr in range(self.config['budget']):
                self.run(belief_node)

class BeliefNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)
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
