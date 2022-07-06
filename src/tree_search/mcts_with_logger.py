from tree_search.mcts import MCTSDPW, DecisionNode

class MCTSDPWLogger(MCTSDPW):
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['GIVEWAY', 'IDLE'],
               6 : ['ABORT', 'IDLE']
               }
    def __init__(self, config=None):
        super(MCTSDPWLogger, self).__init__(config)

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = DecisionNode(parent=None, config=self.config)

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
        print('############### Iter #################')
        while self.not_exit_tree(depth, state_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = state_node.get_child(
                                        self.available_options(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            # try:
            #     print('dec >>> ', self.OPTIONS[decision], '  reward:', reward, \
            #           '  speed:', round(state.sdv.speed, 2), '  ', state.sdv.neighbours['rl'].id)
            # except:
            #     print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward, \
            #           '  speed:', round(state.sdv.speed, 2))

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
            #     print('dec >>> ', self.OPTIONS[decision], '  reward:', reward, \
            #           '  speed:', round(state.sdv.speed, 2), '  ', state.sdv.neighbours['rl'].id)
            # except:
            #     print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward, \
            #           '  speed:', round(state.sdv.speed, 2))

            total_reward += self.config["gamma"] ** rollout_depth * reward
            self.log_visited_sdv_state(state, tree_states, 'rollout')
            self.extract_belief_info(state, rollout_depth)
            # if decision == 4:
                # print('merge')
                # if state.sdv.is_merge_initiated():
                #     print('init')

            if terminal:
                break
        # print('accum reward: ', total_reward)
        return tree_states, total_reward
