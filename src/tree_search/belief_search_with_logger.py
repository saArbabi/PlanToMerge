from tree_search.qmdp import QMDP, BeliefNode

class BeliefSearchLogger(QMDP):
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['GIVEWAY', 'IDLE'],
               6 : ['ABORT', 'IDLE']
               }
    def __init__(self):
        super(BeliefSearchLogger, self).__init__()

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = BeliefNode(parent=None, config=self.config)

    def run(self, belief_node):
        total_reward = 0
        depth = 0
        terminal = False
        state = self.sample_belief(belief_node)

        print('############### Iter #################')
        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.available_options(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            try:
                rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
                print('dec >>> ', self.OPTIONS[decision], \
                      '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
                                            '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
                                            '  rl_params', rl_params, '  ', \
                                            '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
                                            '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            except:
                print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            belief_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)
            self.update_belief(belief_node)
            state = self.sample_belief(belief_node)
            total_reward += self.config["gamma"] ** depth * reward
            depth += 1

        if not terminal:
            total_reward = self.evaluate(state,
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)

    def evaluate(self, state, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        print('############### EVAL #################')
        for rollout_depth in range(depth+1, self.config["horizon"]+1):
            decision = self.rng.choice(self.available_options(state))
            observation, reward, terminal = self.step(state, decision, 'random_rollout')
            try:
                rl_params = [round(val, 2) for val in state.sdv.neighbours['rl'].driver_params.values()]
                print('dec >>> ', self.OPTIONS[decision], \
                      '  reward:', reward, '  rl_id:', state.sdv.neighbours['rl'].id, '  ', \
                                            '  rl_detaXX:', round(state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x), '  ', \
                                            '  rl_params', rl_params, '  ', \
                                            '  rl_att:', state.sdv.neighbours['rl'].neighbours['att'].id, '  ', \
                                            '  rl_act:', round(state.sdv.neighbours['rl'].act_long_c, 2))
            except:
                print('***dec >>> ', self.OPTIONS[decision], '  reward:', reward)

            total_reward += self.config["gamma"] ** rollout_depth * reward

            if terminal:
                break
        # assert 4 == 2, 'ph shit '

        return total_reward
