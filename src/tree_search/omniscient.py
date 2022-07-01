from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.imagined_env import ImaginedEnv

class Omniscient(MCTSDPW):
    def __init__(self, config=None):
        super(Omniscient, self).__init__(config)

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
