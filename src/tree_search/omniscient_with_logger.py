from tree_search.mcts import DecisionNode, ChanceNode
from tree_search.mcts_with_logger import MCTSDPWLogger
from tree_search.imagined_env import ImaginedEnv

class OmniscientLogger(MCTSDPWLogger):
    def __init__(self, config=None):
        super(OmniscientLogger, self).__init__(config)

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

        # self.add_position_noise(state)
        observation = state.planner_observe()
        reward = state.get_reward(decision)
        terminal = state.is_terminal(decision)
        return observation, reward, terminal

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = OmniDecisionNode(parent=None, config=self.config)

    def run(self, state_node):
        """
            Run an iteration of MCTSDPW, starting from a given state
        :param state: the initial environment state
        """
        total_reward = 0
        depth = 0
        terminal = False
        state = state_node.draw_sample(self.rng)
        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}
        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        # print('############### Iter #################')
        while self.not_exit_tree(depth, state_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = state_node.get_child(
                                        self.available_options(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            # try:
            #     rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
            #     print('dec >>> ', self.OPTIONS[decision], \
            #           '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
            #                                 '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
            #                                 '  rl_params', rl_params, '  ', \
            #                                 '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
            #                                 '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            # except:
            #     print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            state_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)

            state = state_node.fetch_state()
            total_reward += self.config["gamma"] ** depth * reward
            depth += 1
            self.log_visited_sdv_state(state, tree_states, 'selection')
            self.extract_belief_info(state, depth)

        if not terminal:
            tree_states, total_reward = self.evaluate(state,
                                         tree_states,
                                          total_reward,
                                          depth)
        # Backup global statistics
        state_node.backup_to_root(total_reward)
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
        # print('############### EVAL #################')
        for rollout_depth in range(depth+1, self.config["horizon"]+1):
            decision = self.rng.choice(self.available_options(state))
            observation, reward, terminal = self.step(state, decision, 'random_rollout')
            # try:
            #     rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
            #     print('dec >>> ', self.OPTIONS[decision], \
            #           '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
            #                                 '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
            #                                 '  rl_params', rl_params, '  ', \
            #                                 '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
            #                                 '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            # except:
            #     print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            total_reward += self.config["gamma"] ** rollout_depth * reward
            self.log_visited_sdv_state(state, tree_states, 'rollout')
            self.extract_belief_info(state, rollout_depth)

            if terminal:
                break
        # print('accum reward: ', total_reward)
        return tree_states, total_reward

class OmniDecisionNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def draw_sample(self, rng):
        """Note: Unlike mcts, here there is no uniform re-sampling of driver parameters.
        """
        img_state = ImaginedEnv(self.state)
        return img_state

class OmniChanceNode(ChanceNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def expand(self, state, obs_id):
        self.children[obs_id] = OmniDecisionNode(self, self.config)
        self.children[obs_id].state = state
