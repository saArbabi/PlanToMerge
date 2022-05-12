import numpy as np
import time
from tree_search.factory import safe_deepcopy_env
from tree_search.abstract import AbstractPlanner, Node
import hashlib
import json

class MCTSDPW(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search with Upper Confidence Tree exploration
       and Double Progressive Widenning.
    """
    # OPTIONS_CAT = {
    #             'LANEKEEP' : [1, 2, 3, 4, 5, 6],
    #             'LANEKEE-ONLY' : [1, 2, 3],
    #             'MERGE' : [4, 5, 6]}
    OPTIONS_CAT = {
                'LANEKEEP' : [1, 4],
                'LANEKEE-ONLY' : [1],
                'MERGE' : [4]}

    def __init__(self):
        """
            New MCTSDPW instance.

        :param config: the mcts configuration. Use default if None.
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        self.config = self.default_config()
        super(MCTSDPW, self).__init__()
        self.reset()

    @classmethod
    def default_config(cls):
        with open('./src/tree_search/config_files/config.json', 'rb') as handle:
            cfg = json.load(handle)
            return cfg

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = DecisionNode(parent=None, config=self.config)

    def get_available_decisions(self, state):
        # return [1, 2, 3, 4, 5, 6]

        # return [2, 5]
        # if not state.sdv.decision:
        if state.sdv.is_merge_possible():
            if not state.sdv.decision or state.sdv.decision == 2:
                return [2, 5]
            elif state.sdv.decision == 5:
                return [5]
        else:
            return [2]

        # elif state.sdv.decision == 2:
        #     return [2]
        # elif state.sdv.decision == 2:
        #     return [2]



        # if state.sdv.decision == 2 or not state.sdv.decision:
        #     return [2, 5]
        # else:
        #     return [5]
        #
        # return [1, 4]
        # if state.sdv.glob_x < state.sdv.merge_lane_start:
        #     return self.OPTIONS_CAT['LANEKEE-ONLY']
        # return self.OPTIONS_CAT[state.sdv.decision_cat]

    def extract_belief_info(self, state, depth):
        vehicle_id = 2
        if depth not in self.belief_info:
            self.belief_info[depth] = {}
            self.belief_info[depth]['xs'] = [veh.glob_x for veh in state.vehicles if veh.id == vehicle_id]
            self.belief_info[depth]['ys'] = [veh.glob_y for veh in state.vehicles if veh.id == vehicle_id]
        else:
            self.belief_info[depth]['xs'].extend([veh.glob_x for veh in state.vehicles if veh.id == vehicle_id])
            self.belief_info[depth]['ys'].extend([veh.glob_y for veh in state.vehicles if veh.id == vehicle_id])

    def extract_tree_info(self, tree_states):
        self.tree_info.append(tree_states)

    def not_exit_tree(self, depth, decision_node, terminal):
        not_exit = depth < self.config['horizon'] and \
                (decision_node.count != 0 or decision_node == self.root) and not terminal
        return not_exit

    def log_visited_sdv_state(self, state, tree_states, mcts_stage):
        """ Use this for visualising tree """
        if mcts_stage == 'selection':
            tree_states['x'].append(state.sdv.glob_x)
            tree_states['y'].append(state.sdv.glob_y)
        elif mcts_stage == 'rollout':
            tree_states['x_rollout'].append(state.sdv.glob_x)
            tree_states['y_rollout'].append(state.sdv.glob_y)

        return tree_states

    def run(self, state, observation):
        """
            Run an iteration of MCTSDPW, starting from a given state
        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        decision_node = self.root
        total_reward = 0
        depth = 0
        state.seed(self.rng.randint(1e5))
        terminal = False
        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}

        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        while self.not_exit_tree(depth, decision_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = decision_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision)
            total_reward += self.config["gamma"] ** depth * reward
            decision_node = chance_node.get_child(
                                            observation,
                                            self.rng)
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
        decision_node.backup_to_root(total_reward)
        self.extract_tree_info(tree_states)

    def evaluate(self, state, tree_states, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """

        self.log_visited_sdv_state(state, tree_states, 'rollout')
        # self.extract_belief_info(state, depth)
        for rollout_depth in range(depth+1, self.config["horizon"]+1):
            decision = self.rng.choice(self.get_available_decisions(state))
            # print('######### ', rollout_depth, ' ########################### in rollout')
            observation, reward, terminal = self.step(state, decision)
            total_reward += self.config["gamma"] ** rollout_depth * reward
            self.log_visited_sdv_state(state, tree_states, 'rollout')
            self.extract_belief_info(state, rollout_depth)

            if terminal:
                break

        return tree_states, total_reward

    def plan(self, state, observation):
        self.reset()
        for plan_itr in range(self.config['budget']):
            self.run(safe_deepcopy_env(state), observation)

    def get_decision(self):
        """Only return the first decision, the rest is conditioned on observations"""
        chosen_decision, self.decision_counts = self.root.selection_rule()
        return chosen_decision

    def is_decision_time(self):
        """The planner computes a decision if certain number of
        timesteps have elapsed from the last decision
        """
        if self.steps_till_next_decision == 0:
            return True

class DecisionNode(Node):
    def __init__(self, parent, config):
        super(DecisionNode, self).__init__(parent)
        self.value = 0
        self.k_decision = config["k_decision"]
        self.alpha_decision = config["alpha_decision"]
        self.temperature = config["temperature"]
        self.config = config

    def unexplored_decisions(self, available_decisions):
        return set(self.children.keys()).symmetric_difference(available_decisions)

    def expand(self, available_decisions, rng):
        decision = rng.choice(list(self.unexplored_decisions(available_decisions)))
        self.children[decision] = ChanceNode(self, self.config)
        return self.children[decision], decision

    def get_child(self, available_decisions, rng):
        if len(self.children) == len(available_decisions) \
                or self.k_decision*self.count**self.alpha_decision < len(self.children):
            # select one of previously expanded decisions
            return self.selection_strategy(rng)
        else:
            # insert a new aciton
            return self.expand(available_decisions, rng)

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.backup_to_root(total_reward)

    def ucb_value(self, decision):
        exploitation = self.children[decision].value
        exploration = np.sqrt(np.log(self.count) / self.children[decision].count)
        ucb_val = exploitation + self.temperature * exploration
        return ucb_val

    def selection_strategy(self, rng):
        """
            Select an decision according to UCB.
        :return: the selected decision with maximum value and exploration bonus.
        """
        decisions = list(self.children.keys())
        indexes = []
        for decision in decisions:
            ucb_val = self.ucb_value(decision)
            indexes.append(ucb_val)

        decision = decisions[self.random_argmax(indexes, rng)]
        return self.children[decision], decision

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        decisions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in decisions])
        decision_counts = {'decisions':decisions,
                                        'counts':[self.children[a].count for a in decisions]}


        return decisions[max(counts, key=(lambda i: self.children[decisions[i]].value))], decision_counts

class ChanceNode(Node):
    def __init__(self, parent, config):
        assert parent is not None
        super().__init__(parent)
        # state progressive widenning parameters
        self.k_state = config["k_state"]
        self.alpha_state = config["alpha_state"]
        self.config = config
        self.value = 0

    def expand(self, obs_id):
        self.children[obs_id] = DecisionNode(self, self.config)

    def get_child(self, observation, rng):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        # print(len(self.children))
        # print(observation)
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = rng.choice(list(self.children))
                return self.children[obs_id]
            else:
                # Add observation to the children set
                self.expand(obs_id)

        return self.children[obs_id]

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        assert self.children
        assert self.parent
        self.update(total_reward)
        self.parent.backup_to_root(total_reward)
