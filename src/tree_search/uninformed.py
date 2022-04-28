from tree_search.mcts import MCTSDPW
import numpy as np

class Uninformed(MCTSDPW):
    def __init__(self):
        super(Uninformed, self).__init__()

    def step(self, state, decision):
        self.belief_estimator(state)
        for i in range(10):
            state.step(decision)

        observation = state.planner_observe()
        reward = state.get_reward()
        terminal = state.is_terminal()
        return observation, reward, terminal

    def belief_estimator(self, state):
        for vehicle in state.vehicles:
            if vehicle.id != 'sdv':
                vehicle.driver_params['aggressiveness'] = np.random.uniform(0.5, 0.9)
                vehicle.set_driver_params()
 
    def run(self, state, observation):
        decision_node = self.root
        total_reward = 0
        depth = 0
        state.seed(self.rng.randint(1e5))
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
