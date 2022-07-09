class RuleBased():
    def __init__(self):
        self.steps_per_decision = 10 # number of timesteps that lapce between each decision
        self.decision_counts = None
        self.politeness_factor = 0.7
        self.initialize_planner()

    def initialize_planner(self):
        self.steps_till_next_decision = 0

    def is_decision_time(self):
        if self.steps_till_next_decision == 0:
            self.steps_till_next_decision = self.steps_per_decision
            return True

    def get_ttm(self, sdv):
        delta_y = sdv.lane_width*1.5 - sdv.glob_y # merger lateral distance to main road
        ttm_m_veh = delta_y/sdv.lateral_actions['move_left'] # steps to merge
        return ttm_m_veh

    def is_bad_state(self, state):
        for vehicle in state.vehicles:
            if state.sdv.lane_decision != 'keep_lane':
                # Too close
                if 0 < (state.sdv.glob_x - vehicle.glob_x) < 3:
                    return True
                # TTC
                if state.sdv.glob_x > vehicle.glob_x:
                    if state.sdv.speed < vehicle.speed:
                        ttc = (state.sdv.glob_x - vehicle.glob_x)/(vehicle.speed - state.sdv.speed)
                        if ttc < 2: # min in dataset is 1.67
                            return True
                elif state.sdv.glob_x < vehicle.glob_x:
                    if state.sdv.speed > vehicle.speed:
                        ttc = (state.sdv.glob_x - vehicle.glob_x)/(vehicle.speed - state.sdv.speed)
                        if ttc < 2: # min in dataset is 1.67
                            return True

    def get_ttc(self, sdv):
        if sdv.speed < sdv.neighbours['rl'].speed:
            ttc = (sdv.glob_x - sdv.neighbours['rl'].glob_x)/(sdv.neighbours['rl'].speed - sdv.speed)
            return ttc

    def available_options(self, state):
        if state.sdv.decision == 6:
            if state.sdv.neighbours['rl']:
                options = [6]
            else:
                options = [4]

        elif state.sdv.decision == 4 and state.sdv.is_merge_initiated():
            options = [4]

        elif state.sdv.is_merge_possible():
            if not state.sdv.neighbours['rl']:
                options = [4]
            else:
                options = [4, 2]
        else:
            options = [2]

        return options

    def get_decision(self, state):
        available_options = self.available_options(state)
        if len(available_options) == 1:
            return available_options[0]

        ttm_m_veh = self.get_ttm(state.sdv)
        gap_to_merge = ttm_m_veh*state.sdv.speed + state.sdv.glob_x-state.sdv.neighbours['rl'].glob_x
        ttm_rl_veh = gap_to_merge/state.sdv.neighbours['rl'].speed

        gap_to_merge = ttm_m_veh*state.sdv.speed + state.sdv.neighbours['fl'].glob_x-state.sdv.glob_x
        ttm_fl_veh = gap_to_merge/state.sdv.neighbours['fl'].speed

        if state.sdv.decision == 4:
            if ttm_fl_veh < ttm_m_veh < ttm_rl_veh*self.politeness_factor:
                return 4
            return 6

        elif state.sdv.decision == 2:
            if ttm_fl_veh < ttm_m_veh < ttm_rl_veh*self.politeness_factor:
                return 4
            else:
                return 2
