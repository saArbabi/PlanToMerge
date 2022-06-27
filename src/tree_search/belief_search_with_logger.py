from tree_search.qmdp import QMDP, BeliefNode

class BeliefSearchLogger(QMDP):
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

        while self.not_exit_tree(depth, belief_node, terminal):
            state = self.sample_belief(belief_node)
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.get_available_decisions(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
            belief_node = chance_node.get_child(
                                            state,
                                            observation,
                                            self.rng)
            self.update_belief(belief_node)
            total_reward += self.config["gamma"] ** depth * reward
            depth += 1

        if not terminal:
            total_reward = self.evaluate(self.sample_belief(belief_node),
                                          total_reward,
                                          depth=depth)
        # Backup global statistics
        belief_node.backup_to_root(total_reward)
