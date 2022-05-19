import numpy as np
import time
from tree_search.factory import safe_deepcopy_env
from tree_search.abstract import AbstractPlanner, Node
from tree_search.imagined_env import ImaginedEnv
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
        self.img_state = ImaginedEnv()
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
        if state.sdv.is_merge_possible():
            if not state.sdv.decision or state.sdv.decision == 2:
                return [2, 5]

            elif state.sdv.is_merge_initiated():
                return [5]

            elif state.sdv.decision == 7:
                if state.sdv.neighbours['rl']:
                    return [7]
                else:
                    return [5, 7]
            elif state.sdv.decision == 5:
                return [5, 7]
        else:
            return [2]


    def predict_vehicle_actions(self, state):
        """
        Returns the joint action of all vehicles other than SDV on the road
        """
        joint_action = []
        for vehicle in state.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(state.all_cars() + \
                                                       [state.dummy_stationary_car])
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

    def not_exit_tree(self, depth, state_node, terminal):
        not_exit = depth < self.config['horizon'] and \
                (state_node.count != 0 or state_node == self.root) and not terminal
        return not_exit

    def imagine_state(self, state):
        """
        Returns an "imagined" environment state, with uniform prior belief.
        """
        self.img_state.copy_attrs(state)
        self.img_state.seed(self.rng.randint(1e5))
        self.img_state.uniform_prior()
        return self.img_state

    def run(self, state_node):
        """
            Run an iteration of MCTSDPW, starting from a given state
        :param state: the initial environment state
        """
        total_reward = 0
        depth = 0
        state = self.get_env_state(state_node)
        terminal = False

        while self.not_exit_tree(depth, state_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = state_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision)
            child_type, state_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)

            state = self.get_env_state(state_node)
            if child_type == 'old':
                reward = state_node.state.get_reward()

            total_reward += self.config["gamma"] ** depth * reward
            depth += 1

        if not terminal:
            total_reward = self.evaluate(state,
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        state_node.backup_to_root(total_reward)

    def evaluate(self, state, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for rollout_depth in range(depth+1, self.config["horizon"]+1):
            decision = self.rng.choice(self.get_available_decisions(state))
            observation, reward, terminal = self.step(state, decision)
            total_reward += self.config["gamma"] ** rollout_depth * reward
            if terminal:
                break

        return total_reward

    def get_env_state(self, state_node):
        return state_node.state.copy_this_state()

    def plan(self, state):
        self.reset()
        state_node = self.root
        for plan_itr in range(self.config['budget']):
            state_node.state = self.imagine_state(state)
            self.run(state_node)

    def get_decision(self):
        """Only return the first decision, the rest is conditioned on observations"""
        chosen_decision, self.decision_counts = self.root.selection_rule()
        return chosen_decision

    def is_decision_time(self):
        """The planner computes a decision if certain number of
        timesteps have elapsed from the last decision
        """
        if self.steps_till_next_decision == 0:
            self.steps_till_next_decision = self.steps_per_decision
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

    def expand(self, state, obs_id):
        self.children[obs_id] = DecisionNode(self, self.config)
        self.children[obs_id].state = state

    def get_child(self, state, observation, rng):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        # print(len(self.children))
        # print(observation)
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = rng.choice(list(self.children))
                child_type = 'old'
                return child_type, self.children[obs_id]
            else:
                # Add observation to the children set
                child_type = 'new'
                self.expand(state, obs_id)
                return child_type, self.children[obs_id]
        else:
            child_type = 'old'
            return child_type, self.children[obs_id]

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        assert self.children
        assert self.parent
        self.update(total_reward)
        self.parent.backup_to_root(total_reward)
