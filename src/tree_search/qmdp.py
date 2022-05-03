from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.abstract import AbstractPlanner, Node
from tree_search.factory import safe_deepcopy_env
import time
import hashlib
import numpy as np
import sys
from tree_search.imagined_env import ImaginedEnv
from tree_search.belief_net import BeliefNet

class QMDP(MCTSDPW):
    def __init__(self):
        super(QMDP, self).__init__()
        self.update_counts = 0
        self._enough_history = False
        self.decision_counts = False
        self.nidm = BeliefNet()
        self.img_state = ImaginedEnv()

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = BeliefNode(parent=None, config=self.config)

    def is_decision_time(self, env):
        """The planner computes a decision if:
            1) enough history observations are collected
            2) certain number of timesteps have elapsed from the last decision
        """
        if self.enough_history(env):
            if self.steps_till_next_decision == 0:
                self.steps_till_next_decision = self.steps_per_decision
                return True
            else:
                self.steps_till_next_decision -= 1

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

    def get_joint_action(self, state):
        """
        Returns the joint action of all vehicles other than SDV on the road
        """
        joint_action = []
        for vehicle in state.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(state.all_cars() + \
                                                       [state.dummy_stationary_car])


            actions = vehicle.act()
            # if vehicle.id == 2:
            #     actions = vehicle.act()
            #
            # else:
            #     actions = self.nidm.estimate_vehicle_action(vehicle)


            joint_action.append(actions)
            act_long = actions[0]

            if act_long < -5:
                state.is_large_deceleration = True

            if state.time_step > 0:
                vehicle.act_long_p = vehicle.act_long_c
            vehicle.act_long_c = act_long


        return joint_action
    def step(self, state, decision):
        state.env_reward_reset()
        for i in range(self.steps_per_decision):
            joint_action = self.get_joint_action(state)
            state.step(joint_action, decision)
            ##############
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
        # print(total_reward)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)
        self.extract_tree_info(tree_states)

    def update_belief(self, belief_node, state):
        """
        Passes the sequence of past vehicle observations (vehicle viewpoint)
        and actions into an LSTM encoder. Then the encoded state is mapped to
        the latent belief.
        """
        if not belief_node.belief_params:
            # self.planner.update_counts += 1
            belief_node.belief_params = 1
            belief_node.state = state
        self.nidm.latent_inference(state.vehicles)

    def imagine_state(self, state):
        """
        Returns an "imagined" environment.
        """
        self.img_state.copy_attrs(state)
        return self.img_state

    def plan(self, state, observation):
        """
        need observation?
        """
        self.reset()
        state = self.imagine_state(state)
        belief_node = self.root
        self.update_belief(belief_node, state)
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
        self.belief_params = None # this holds the belief parameters

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




                # vehicle.mu = vehicle.driver_params['aggressiveness']
                # vehicle.var = 0.2
            # print('id ', vehicle.id)
            # print('vehicle.mu ', vehicle.mu)
        # self.belief_params = [mu, var]
        # self.state = [state, ]


    def sample_belief(self):
        """
        Returns a sample from the belief state
        """
        sampled_state = safe_deepcopy_env(self.state)
        # for vehicle in sampled_state.vehicles:
        #     vehicle.driver_params['aggressiveness'] = 0.05
        #     vehicle.set_driver_params()
        return sampled_state

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
