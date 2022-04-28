from tree_search.mcts import MCTSDPW

class Omniscient(MCTSDPW):
    def __init__(self):
        super(Omniscience, self).__init__()

    def run(self, state, observation):
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
