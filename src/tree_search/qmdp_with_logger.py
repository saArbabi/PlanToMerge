from tree_search.qmdp import QMDP, BeliefNode

class QMDPLogger(QMDP):
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['GIVEWAY', 'IDLE'],
               6 : ['ABORT', 'IDLE']
               }
    def __init__(self):
        super(QMDPLogger, self).__init__()

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = BeliefNode(parent=None, config=self.config)

    def log_visited_sdv_state(self, state, tree_states, mcts_stage):
        """ Use this for visualising tree """
        if mcts_stage == 'selection':
            tree_states['x'].append(state.sdv.glob_x)
            tree_states['y'].append(state.sdv.glob_y)
        elif mcts_stage == 'rollout':
            tree_states['x_rollout'].append(state.sdv.glob_x)
            tree_states['y_rollout'].append(state.sdv.glob_y)
        return tree_states

    def extract_belief_info(self, state, depth):
        for veh in state.vehicles:
            if veh.id not in self.belief_info:
                self.belief_info[veh.id] = {}

            if depth not in self.belief_info[veh.id]:
                self.belief_info[veh.id][depth] = {}
                self.belief_info[veh.id][depth]['xs'] = []
                self.belief_info[veh.id][depth]['ys'] = []

            self.belief_info[veh.id][depth]['xs'].append(veh.glob_x)
            self.belief_info[veh.id][depth]['ys'].append(veh.glob_y)

    def extract_tree_info(self, tree_states):
        self.tree_info.append(tree_states)

    def run(self, belief_node):
        """
        Note: QMDP only holds belief in the first tree node
        """
        total_reward = 0
        depth = 0
        terminal = False
        state = self.sample_belief(belief_node)

        tree_states = {
                        'x':[], 'y':[],
                        'x_rollout':[], 'y_rollout':[]}
        self.extract_belief_info(state, 0)
        self.log_visited_sdv_state(state, tree_states, 'selection')
        # print('############### Iter #################')
        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.available_options(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            # if decision != 44:
            #     try:
            #         rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
            #         print('dec >>> ', self.OPTIONS[decision], \
            #               '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
            #                                     '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
            #                                     '  rl_params', rl_params, '  ', \
            #                                     '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
            #                                     '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            #     except:
            #         print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            belief_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)

            state = belief_node.fetch_state()
            total_reward += self.config["gamma"] ** depth * reward
            depth += 1

            self.log_visited_sdv_state(state, tree_states, 'selection')
            self.extract_belief_info(state, depth)

        if not terminal:
            tree_states, total_reward = self.evaluate(state,
                                         tree_states,
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)
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
            # if decision != 44:
            #     try:
            #         rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
            #         print('dec >>> ', self.OPTIONS[decision], \
            #               '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
            #                                     '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
            #                                     '  rl_params', rl_params, '  ', \
            #                                     '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
            #                                     '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            #     except:
            #         print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            total_reward += self.config["gamma"] ** rollout_depth * reward
            self.log_visited_sdv_state(state, tree_states, 'rollout')
            self.extract_belief_info(state, rollout_depth)

            if terminal:
                break
        # assert 4 == 2, 'ph shit '

        return tree_states, total_reward
