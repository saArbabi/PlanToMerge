# import gym
# from gym.utils import seeding
import numpy as np

class AbstractPlanner(object):
    def __init__(self):
        self.root = None
        self.steps_till_next_decision = 0
        self.steps_per_decision = 10 # number of timesteps that lapce between each decision

    def seed(self, seed):
        """
            Seed the planner randomness source, e.g. for rollout policy
        :param seed: the seed to be used
        """
        self.rng = np.random.RandomState(seed)

    def plan(self, state, observation):
        """
            Plan an optimal sequence of decisions.

        :param state: the initial environment state
        :param observation: the corresponding state observation
        :return: the decisions sequence
        """
        raise NotImplementedError()

    def get_plan(self):
        """
            Get the optimal decision sequence of the current tree by recursively selecting the best decision within each
            node with no exploration.

        :return: the list of decisions
        """
        decisions = []
        node = self.root
        while node.children:
            decision, _ = node.selection_rule()
            decisions.append(decision)
            node = node.children[decision]
        return decisions

    def step(self, state, decision):
        state.env_reward_reset()
        for i in range(self.steps_per_decision):
            state.step(decision)

        observation = state.planner_observe()
        reward = state.get_reward(decision)
        terminal = state.is_terminal()
        return observation, reward, terminal

    def reset(self):
        raise NotImplementedError

class Node(object):
    """
        A tree node
    """

    def __init__(self, parent):
        """
            New node.
        :param parent: its parent node
        """
        self.parent = parent
        self.children = {}
        """ Dict of children nodes, indexed by decision labels"""

        self.count = 0
        """ Number of times the node was visited."""

        self.value_upper = 0
        """ Estimated value of the node's decision sequence"""

    def selection_rule(self):
        raise NotImplementedError()

    def is_leaf(self):
        return not self.children

    def path(self):
        """
        :return: sequence of decision labels from the root to the node
        """
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path)

    def sequence(self):
        """
        :return: sequence of nodes from the root to the node
        """
        node = self
        path = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return reversed(path)

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += (total_reward - self.value)/self.count

    @staticmethod
    def all_argmax(x):
        """
        :param x: a set
        :return: the list of indexes of all maximums of x
        """
        m = np.amax(x)
        return np.nonzero(x == m)[0]

    def random_argmax(self, x, rng):
        """
            Randomly tie-breaking arg max
        :param x: an array
        :return: a random index among the maximums
        """
        indices = Node.all_argmax(x)
        return rng.choice(indices)

    def __str__(self):
        return "{} (n:{}, v:{:.2f})".format(list(self.path()), self.count, self.value)

    def __repr__(self):
        return '<node {}>'.format(id(self))
