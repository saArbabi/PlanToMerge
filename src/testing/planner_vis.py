import sys
sys.path.insert(0, './src')
from vis.plan_viewer import Viewer
from vis.tree_vis import TreeVis
from envs.auto_merge import EnvAutoMerge
import matplotlib.pyplot as plt
import numpy as np
import json

def load_planner():
    with open('./src/tree_search/config_files/config.json', 'rb') as handle:
        cfg = json.load(handle)
        planner_type = cfg['planner_type']

    if planner_type == 'uninformed':
        from tree_search.uninformed import Uninformed
        planner = Uninformed()

    if planner_type == 'omniscient':
        from tree_search.omniscient import Omniscient
        planner = Omniscient()

    if planner_type == 'mcts':
        from tree_search.mcts import MCTSDPW
        planner = MCTSDPW()

    if planner_type == 'belief_search':
        from tree_search.belief_search import BeliefSearch
        planner = BeliefSearch()
    return planner

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvAutoMerge(config)
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
                vis_tree.save_tree_snapshot(env.sdv.planner, env.time_step)
            try:
                viewer.focus_on_this_vehicle = user_input
            except:
                pass


        obs = env.planner_observe()
        if env.sdv.is_decision_time():
            planner.plan(env, obs)
            decision = planner.get_decision()
        else:
            decision = env.sdv.decision


        all_cars = env.all_cars()
        viewer.log_var(all_cars)
        viewer.render(all_cars, planner)
        env.step(decision)

if __name__=='__main__':
    main()
