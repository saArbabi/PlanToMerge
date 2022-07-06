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
    def __init__(self, config=None):
        """
            New MCTSDPW instance.

        :param config: the mcts configuration. Use default if None.
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        if not config:
            self.config = self.default_config()
            print('Default planner params: ')
            print(self.config)
        else:
            self.config = config
        self.decision_counts = None
        super(MCTSDPW, self).__init__()

    @classmethod
    def default_config(cls):
        with open('./src/tree_search/config_files/config.json', 'rb') as handle:
            cfg = json.load(handle)
            return cfg

    def reset(self):
        self.root = DecisionNode(parent=None, config=self.config)

    def get_available_decisions(self, state):
        if state.sdv.decision == 5:
            if state.sdv.neighbours['rl']:
                return [5]
            else:
                return [4]
        else:
            if state.sdv.is_merge_initiated():
                return [4]
            elif state.sdv.decision_cat == 'LANEKEEP':
                if state.sdv.is_merge_possible():
                    if not state.sdv.neighbours['rl']:
                        return [4]
                    if state.sdv.driver_params['aggressiveness'] == 0:
                        return [1, 2, 4, 5]
                    elif state.sdv.driver_params['aggressiveness'] == 1:
                        return [3, 2, 4, 5]
                    else:
                        return [1, 2, 3, 4, 5]
                else:
                    if state.sdv.driver_params['aggressiveness'] == 0:
                        return [1, 2]
                    elif state.sdv.driver_params['aggressiveness'] == 1:
                        return [3, 2]
                    else:
                        return [1, 2, 3]
            elif state.sdv.decision_cat == 'MERGE':
                return [4, 5]

    def predict_joint_action(self, state):
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

    def add_position_noise(self, state):
        for vehicle in state.vehicles:
            vehicle.glob_x += self.rng.normal()

    def step(self, state, decision, step_type):
        state.env_reward_reset()
        state.sdv.update_decision(decision)
        if step_type == 'search':
            for i in range(self.steps_per_decision):
                joint_action = self.predict_joint_action(state)
                state.step(joint_action)

        elif step_type == 'random_rollout':
            for i in range(self.steps_per_decision):
                if i % self.config['rollout_step_skips'] == 0:
                    joint_action = self.predict_joint_action(state)
                state.step(joint_action)

        self.add_position_noise(state)
        observation = state.planner_observe()
        reward = state.get_reward(decision)
        terminal = state.is_terminal(decision)
        return observation, reward, terminal

    def not_exit_tree(self, depth, state_node, terminal):
        not_exit = depth < self.config['horizon'] and \
                (state_node.count != 0 or state_node == self.root) and not terminal
        return not_exit

    def run(self, state_node):
        """
            Run an iteration of MCTSDPW, starting from a given state
        :param state: the initial environment state
        """
        total_reward = 0
        depth = 0
        terminal = False
        state = state_node.draw_sample(self.rng)

        while self.not_exit_tree(depth, state_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = state_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            state_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)

            state = state_node.fetch_state()
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
            state.env_reward_reset()
            state_before = safe_deepcopy_env(state)
            observation, reward, terminal = self.step(state, decision, 'random_rollout')
            total_reward += self.config["gamma"] ** rollout_depth * reward

            if terminal:
                break
        return total_reward

    def plan(self, state):
        available_decisions = self.get_available_decisions(state)
        if len(available_decisions) > 1:
            self.reset()
            state_node = self.root
            state_node.state = ImaginedEnv(state)
            for plan_itr in range(self.config['budget']):
                self.run(state_node)

    def get_decision(self, state):
        """Only return the first decision, the rest is conditioned on observations"""
        available_decisions = self.get_available_decisions(state)
        if len(available_decisions) == 1:
            return available_decisions[0]
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

    def fetch_state(self):
        assert self.state, 'This node has no state attribute'
        img_state = ImaginedEnv(self.state)
        return img_state

    def draw_sample(self, rng):
        img_state = ImaginedEnv(self.state)
        img_state.uniform_prior(img_state.vehicles, rng.randint(1e5))
        return img_state

class ChanceNode(Node):
    def __init__(self, parent, config):
        assert parent is not None
        super().__init__(parent)
        # state progressisve widenning parameters
        self.k_state = config["k_state"]
        self.alpha_state = config["alpha_state"]
        self.config = config
        self.value = 0

    def expand(self, state, obs_id):
        self.children[obs_id] = DecisionNode(self, self.config)
        self.children[obs_id].state = state

    def get_child(self, state, observation, rng):
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = rng.choice(list(self.children))
                return self.children[obs_id]
            else:
                # Add observation to the children set
                self.expand(state, obs_id)
                return self.children[obs_id]
        else:
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
