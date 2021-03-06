from tree_search.qmdp import QMDP, BeliefNode

class BeliefSearch(QMDP):
    def __init__(self, config=None):
        super(BeliefSearch, self).__init__(config)

    def run(self, belief_node):
        total_reward = 0
        depth = 0
        terminal = False
        state = self.sample_belief(belief_node)
        
        while self.not_exit_tree(depth, belief_node, terminal):
            # perform a decision followed by a transition
            chance_node, decision = belief_node.get_child(
                                        self.available_options(state),
                                        self.rng)

            observation, reward, terminal = self.step(state, decision, 'search')
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
