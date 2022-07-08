from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.imagined_env import ImaginedEnv

class MCTSMEAN(MCTSDPW):
    def __init__(self, config=None):
        super(MCTSMEAN, self).__init__(config)

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

        self.add_position_noise(state)
        observation = state.planner_observe()
        reward = state.get_reward(decision)
        terminal = state.is_terminal(decision)
        return observation, reward, terminal

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = MEANDecisionNode(parent=None, config=self.config)

class MEANDecisionNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def draw_sample(self, rng):
        img_state = ImaginedEnv(self.state)
        img_state.mean_prior(img_state.vehicles, rng.randint(1e5))
        return img_state

class MEANChanceNode(ChanceNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def expand(self, state, obs_id):
        self.children[obs_id] = MEANDecisionNode(self, self.config)
        self.children[obs_id].state = state
