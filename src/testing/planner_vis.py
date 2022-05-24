import sys
sys.path.insert(0, './src')
from vis.plan_viewer import Viewer
from vis.tree_vis import TreeVis
from envs.auto_merge import EnvAutoMerge
import matplotlib.pyplot as plt
import numpy as np
import time
import json
planner_name = 'mcts'

def load_planner():
    with open('./src/tree_search/config_files/config.json', 'rb') as handle:
        cfg = json.load(handle)
        print('### Planner name is: ', planner_name)

    if planner_name == 'uninformed':
        from tree_search.uninformed import Uninformed
        planner = Uninformed()

    if planner_name == 'omniscient':
        from tree_search.omniscient import Omniscient
        planner = Omniscient()

    if planner_name == 'mcts':
        from tree_search.mcts_with_logger import MCTSDPWLogger
        planner = MCTSDPWLogger()

    if planner_name == 'belief_search':
        from tree_search.belief_search import BeliefSearch
        planner = BeliefSearch()

    if planner_name == 'qmdp':
        from tree_search.qmdp_with_logger import QMDPLogger
        planner = QMDPLogger()

    return planner

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvAutoMerge()
    episode_id = 3
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    vis_tree = TreeVis()
    planner = load_planner()
    while True:
        user_input = input(str(env.time_step) + \
                           ' Enter to continue, n to exit, s to save tree  ')
        if user_input:
            if user_input == 'n':
                sys.exit()
            if user_input == 's':
                vis_tree.save_tree_snapshot(planner, env.time_step)
            try:
                viewer.focus_on_this_vehicle = user_input
            except:
                pass

        if planner.is_decision_time():
            t_0 = time.time()
            planner.plan(env)
            _decision = planner.get_decision()
            avg_reward = planner.get_avg_reward(_decision)
            t_1 = time.time()
            print('compute time: ', t_1 - t_0)
            print('Avg reward: ', avg_reward)
            decision = input(' give me a decision ')
            try:
                decision = int(decision)
            except:
                decision = _decision
            env.sdv.update_decision(decision)

        all_cars = env.all_cars()
        viewer.log_var(all_cars)
        viewer.render(all_cars, planner)

        env.step()
        planner.steps_till_next_decision -= 1

if __name__=='__main__':
    main()
