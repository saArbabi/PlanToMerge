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
        self.root = DecisionNode(parent=None, planner=self)

    def get_available_decisions(self, state):
        """
        Available agent decisions is conditioned on,
        (1) agent state
        (2) last agent decision
        """
        # return [1, 2, 3, 4, 5, 6]
        if state.sdv.glob_x < state.sdv.merge_lane_start:
            return self.OPTIONS_CAT['LANEKEE-ONLY']
        return self.OPTIONS_CAT[state.sdv.decision_cat]

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
        # state.seed(self.rng.randint(1e5))
        terminal = False
        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}

        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        while self.not_exit_tree(depth, decision_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = decision_node.get_child(state, temperature=self.config['temperature'])
            # print('######### ', depth, ' ########################### in tree')
            observation, reward, terminal = self.step(state, decision)
            total_reward += self.config["gamma"] ** depth * reward
            decision_node = chance_node.get_child(observation)
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
            total_reward += self.config["gamma"] ** (rollout_depth) * reward
            self.log_visited_sdv_state(state, tree_states, 'rollout')
            self.extract_belief_info(state, rollout_depth)

            if terminal:
                break

        return tree_states, total_reward

    def plan(self, state, observation):
        self.reset()

        for i in range(self.config['budget']):
            # t0 = time.time()
            self.run(safe_deepcopy_env(state), observation)

            # print('########### run: ', i)
            # for key, d in self.root.children.items():
            #     print('dec: ', key , ' dec_value: ', d.value)
            #     print('dec: ', key , ' dec_count: ', d.count)
            #     print('dec: ', key , ' ucb: ', self.root.ucb_value(key, self.config['temperature']))
            #     if key == 0:
            #         print(self.root.count)
            #         print(self.root.children[key].count)
        node = self.root
        i = 0
        # print(self.root.count)
        # print(self.root.children[0])
        # print(self.root.children[3])
        # # print(self.root.children[3].children.values()[0])
        # print(list(self.root.children[3].children.values())[0])
        # print(list(self.root.children[3].children.values())[1])
        # print(list(self.root.children[3].children.values())[2])
        # while node.children:
        #     print(i)
        #     i += 1
        #     node = node.children[0]

        # print(self.get_plan())

    def get_decision(self):
        """Only return the first decision, the rest is conditioned on observations"""
        chosen_decision, decision_counts = self.root.selection_rule()
        return chosen_decision, decision_counts

class DecisionNode(Node):
    def __init__(self, parent, planner):
        super(DecisionNode, self).__init__(parent, planner)
        self.value = 0
        self.k_decision = self.planner.config["k_decision"]
        self.alpha_decision = self.planner.config["alpha_decision"]

    def unexplored_decisions(self, state):
        if state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            decisions = self.planner.get_available_decisions(state)
        except AttributeError:
            decisions = self.planner.get_available_decisions(state)
        return set(self.children.keys()).symmetric_difference(decisions)

    def expand(self, state):
        # decision = self.planner.np_random.choice(list(self.unexplored_decisions(state)))
        decision = self.planner.rng.choice(list(self.unexplored_decisions(state)))
        self.children[decision] = ChanceNode(self, self.planner)
        return self.children[decision], decision

    def get_child(self, state, temperature=None):
        if len(self.children) == len(self.planner.get_available_decisions(state)) \
                or self.k_decision*self.count**self.alpha_decision < len(self.children):
            # select one of previously expanded decisions
            return self.selection_strategy(temperature)
        else:
            # insert a new aciton
            return self.expand(state)

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.backup_to_root(total_reward)

    def ucb_value(self, decision, temperature):
        exploitation = self.children[decision].value
        exploration = np.sqrt(np.log(self.count) / self.children[decision].count)
        ucb_val = exploitation + temperature * exploration
        # ucb_val = exploitation
        # print('###############')
        # print('exploitation ', exploitation)
        # print('exploration ', exploration)
        return ucb_val

        #
        # # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        # return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def selection_strategy(self, temperature):
        """
            Select an decision according to UCB.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected decision with maximum value and exploration bonus.
        """
        decisions = list(self.children.keys())
        indexes = []
        for decision in decisions:
            ucb_val = self.ucb_value(decision, temperature)
            indexes.append(ucb_val)

        decision = decisions[self.random_argmax(indexes)]
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
    K = 1.0
    """ The value function first-order filter gain"""
    def __init__(self, parent, planner):
        assert parent is not None
        super().__init__(parent, planner)
        # state progressive widenning parameters
        self.k_state = self.planner.config["k_state"]
        self.alpha_state = self.planner.config["alpha_state"]
        self.value = 0

    def expand(self, obs_id):
        self.children[obs_id] = DecisionNode(self, self.planner)

    def get_child(self, observation):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        # print(len(self.children))
        # print(observation)
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = self.planner.rng.choice(list(self.children))
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
