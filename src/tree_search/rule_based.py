class RuleBased():
    def __init__(self):
        self.steps_per_decision = 10 # number of timesteps that lapce between each decision
        self.decision_counts = None
        self.initialize_planner()

    def initialize_planner(self):
        self.steps_till_next_decision = 0

    def is_decision_time(self):
        if self.steps_till_next_decision == 0:
            self.steps_till_next_decision = self.steps_per_decision
            return True

    def get_ttc(self, sdv):
        ttc = (sdv.glob_x - sdv.neighbours['rl'].glob_x)/(sdv.neighbours['rl'].speed - sdv.speed)
        return ttc

    def get_tiv(self, sdv):
        tiv = (sdv.glob_x - sdv.neighbours['rl'].glob_x)/(sdv.neighbours['rl'].speed)
        return tiv


    def available_options(self, state):
        if state.sdv.decision == 4:
            if not state.sdv.neighbours['rl'] or state.sdv.is_merge_initiated():
                options = [4]
            else:
                options = [4, 5]

        elif state.sdv.decision == 5 and \
                state.sdv.neighbours['rl'] and \
                    state.sdv.prev_rl_veh.id >= state.sdv.neighbours['rl'].id:

            options = [5]
        elif state.sdv.is_merge_possible():
            if not state.sdv.neighbours['rl']:
                options = [4]
            elif state.sdv.neighbours['rl']:
                options = [2, 4]
        else:
            options = [2]
        return options

    def get_decision(self, state):
        # print('######')
        available_options = self.available_options(state)
        # print('available_options   :', available_options)
        if len(available_options) == 1:
            return available_options[0]
        ttc = self.get_ttc(state.sdv)
        tiv = self.get_tiv(state.sdv)
        # print('ttc   ', ttc)
        # print('tiv   ', tiv)
        if state.sdv.decision == 4 or state.sdv.decision == 5:
            if (ttc > 5.9 or ttc < 0) and tiv > 2.7:
                return 4
            return 5

        elif state.sdv.decision == 2:
            if (ttc > 5.9 or ttc < 0) and tiv > 2.7:
                return 4
            else:
                return 2
