"""
Run this to visualise a driving episode. The input keys are:
- st: save tree state (with received observations and graphviz nodes)
- slogs: save the true environmnet logs sofar
- slatent: save the encoded history - used in latent plot
- agg: make vehicle 3 aggressive
"""


import sys
sys.path.insert(0, './src')
from vis.plan_viewer import Viewer
from vis.tree_vis import TreeVis
from envs.auto_merge import EnvAutoMerge
import matplotlib.pyplot as plt
import numpy as np
import time
import json
planner_name = 'mcts_mean'
# planner_name = 'mcts'
# planner_name = 'qmdp'
planner_name = 'belief_search'
# planner_name = 'omniscient'
# planner_name = 'rule_based'

def load_planner():
    with open('./src/tree_search/config_files/config.json', 'rb') as handle:
        cfg = json.load(handle)
        print('### Planner name is: ', planner_name)

    if planner_name == 'uninformed':
        from tree_search.uninformed import Uninformed
        planner = Uninformed()

    if planner_name == 'omniscient':
        # from tree_search.omniscient import Omniscient
        # planner = Omniscient()
        from tree_search.omniscient_with_logger import OmniscientLogger
        planner = OmniscientLogger()

    if planner_name == 'mcts':
        from tree_search.mcts_with_logger import MCTSDPWLogger
        planner = MCTSDPWLogger()

    if planner_name == 'mcts_mean':
        from tree_search.mcts_mean import MCTSMEAN
        planner = MCTSMEAN()

    if planner_name == 'belief_search':
        from tree_search.belief_search_with_logger import BeliefSearchLogger
        planner = BeliefSearchLogger()
        # from tree_search.belief_search import BeliefSearch
        # planner = BeliefSearch()

    if planner_name == 'qmdp':
        from tree_search.qmdp_with_logger import QMDPLogger
        planner = QMDPLogger()

    if planner_name == 'rule_based':
        from tree_search.rule_based import RuleBased
        planner = RuleBased()

    return planner

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvAutoMerge()
    episode_id = 557
    episode_id = 511
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    vis_tree = TreeVis()
    planner = load_planner()
    if type(planner).__name__ == 'RuleBased':
        env.sdv.driver_params['aggressiveness'] = 0.5
        env.sdv.set_driver_params()

    decision = 2
    env.sdv.update_decision(decision)
    cumulative_reward = 0
    avg_step_reward = 0
    avg_step_rewards = []
    avg_step_reward_steps = []
    for i in range(5):
        # gather at least 5 obs prior to making a decision
        env.step()

    while True:
        user_input = input(str(env.time_step) + \
                           ' Enter to continue, n to exit, s to save tree  ')
        if user_input:
            if user_input == 'n':
                sys.exit()
            if user_input == 's':
                vis_tree.save_tree_snapshot(planner, env.time_step)
            if user_input == 'st':
                viewer.save_tree_state(planner, last_decision_time_step)
            if user_input == 'slogs':
                viewer.save_state_logs()
            if user_input == 'slatent':
                viewer.save_latent(planner)
            if user_input == 'agg':
                for vehicle in env.vehicles:
                    if vehicle.id == 3:
                        vehicle.driver_params['aggressiveness'] = 0.9
                        vehicle.set_driver_params(np.random.RandomState(episode_id))
            try:
                viewer.focus_on_this_vehicle = user_input
            except:
                pass

        if planner.is_decision_time():
            print('        decision: ', decision)
            avg_step_reward = env.get_reward(decision)
            cumulative_reward += avg_step_reward
            avg_step_rewards.append(avg_step_reward)
            avg_step_reward_steps.append(env.time_step)
            last_decision_time_step = env.time_step
            print('Avg step reward: ', avg_step_reward)
            print('Cummulative reward: ', cumulative_reward)

            env.env_reward_reset()
            t_0 = time.time()
            # planner.plan(env)
            _decision = planner.get_decision(env)
            t_1 = time.time()
            compute_time = t_1 - t_0
            print('compute time: ', compute_time)
            viewer.render_plans(planner)
            decision = input('Give me a decision ')
            try:
                decision = int(decision)
            except:
                decision = _decision
            env.sdv.update_decision(decision)
            print('DECISION: ', decision)

        env.step()
        viewer.render_env(env.all_cars())
        viewer.log_state(env)
        viewer.draw_var_veh = env.sdv.neighbours['rl']
        viewer.render_logs(avg_step_reward_steps, avg_step_rewards)
        planner.steps_till_next_decision -= 1

if __name__=='__main__':
    main()
