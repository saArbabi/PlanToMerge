from tree_search.mcts import MCTSDPW
from tree_search.mcts import MCTSDPW, DecisionNode, ChanceNode
from tree_search.imagined_env import ImaginedEnv

class Omniscient(MCTSDPW):
    def __init__(self, config=None):
        super(Omniscient, self).__init__(config)

    def reset(self):
        self.tree_info = []
        self.belief_info = {}
        self.root = OmniDecisionNode(parent=None, config=self.config)

class OmniDecisionNode(DecisionNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def draw_sample(self, rng):
        """Note: Unlike mcts, here there is no uniform re-sampling of driver parameters.
            There is however stochasticity in the state transisions (action noise). 
        """
        img_state = ImaginedEnv(self.state)
        return img_state

class OmniChanceNode(ChanceNode):
    def __init__(self, parent, config):
        super().__init__(parent, config)

    def expand(self, state, obs_id):
        self.children[obs_id] = OmniDecisionNode(self, self.config)
        self.children[obs_id].state = state
