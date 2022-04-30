from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.abstract import AbstractPlanner, Node
from tree_search.factory import safe_deepcopy_env
import time
import hashlib
import numpy as np
import sys

class BeliefSearch(MCTSDPW):
    def __init__(self):
        super(BeliefSearch, self).__init__()
        self.update_counts = 0

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = BeliefNode(parent=None, planner=self)

    def step(self, state, decision):

        for i in range(10):
            state.step(decision)

        observation = state.planner_observe()
        reward = state.get_reward()
        terminal = state.is_terminal()
        return observation, reward, terminal

    def run(self, belief_node):
        """
        ?
        """
        total_reward = 0
        depth = 0
        # state.seed(self.rng.randint(1e5))
        terminal = False
        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}


        state = belief_node.sample_belief()
        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(state, temperature=self.config['temperature'])
            observation, reward, terminal = self.step(state, decision)
            belief_node = chance_node.get_child(observation)
            belief_node.update_belief(state)
            state = belief_node.sample_belief()

            total_reward += self.config["gamma"] ** depth * reward
            depth += 1
            self.log_visited_sdv_state(state, tree_states, 'selection')
            self.extract_belief_info(state, depth)

        if not terminal:
            tree_states, total_reward = self.evaluate(state,
                                         tree_states,
                                          total_reward,
                                          depth=depth)
        # print(total_reward)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)
        self.extract_tree_info(tree_states)

    def plan(self, state, observation):
        """
        need observation?
        """
        self.reset()
        belief_node = self.root
        belief_node.update_belief(state)
        for plan_itr in range(self.config['budget']):
            # input('This is plan iteration '+ str(plan_itr))
            # t_0 = time.time()
            # t_1 = time.time()
            # print('copy time: ', t_1 - t_0)

            self.run(belief_node)
            # print(self.update_counts)



class BeliefNode(DecisionNode):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)
        self.belief_params = None # this holds the belief parameters

    def expand(self, state):
        decision = self.planner.rng.choice(list(self.unexplored_decisions(state)))
        self.children[decision] = SubChanceNode(self, self.planner)
        return self.children[decision], decision

    def get_child(self, state, temperature=None):
        if len(self.children) == len(self.planner.get_available_decisions(state)) \
                or self.k_decision*self.count**self.alpha_decision < len(self.children):
            # select one of previously expanded decisions
            return self.selection_strategy(temperature)
        else:
            # insert a new aciton
            return self.expand(state)

    def update_belief(self, state):
        if not self.belief_params:
            # self.planner.update_counts += 1
            self.belief_params = 1

        for vehicle in state.vehicles:
            vehicle.mu = vehicle.driver_params['aggressiveness']
            vehicle.var = 0.2
            # print('id ', vehicle.id)
            # print('vehicle.mu ', vehicle.mu)
        # self.belief_params = [mu, var]
        # self.state = [state, ]
        self.state = safe_deepcopy_env(state)


    def sample_belief(self):
        """
        Returns a sample from the belief state
        """
        return self.state

class SubChanceNode(ChanceNode):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)

    def get_child(self, observation):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = self.planner.rng.choice(list(self.children))
                return self.children[obs_id]
            else:
                # Add observation to the children set
                self.expand(obs_id)

        return self.children[obs_id]

    def expand(self, obs_id):
        belief_node = BeliefNode(self, self.planner)
        self.children[obs_id] = belief_node
