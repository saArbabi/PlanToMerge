import sys
sys.path.insert(0, './src')
from plan_viewer import Viewer
from tree_vis import TreeVis
from envs.auto_merge import EnvAutoMerge
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvAutoMerge(config)
    episode_id = 3
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    vis_tree = TreeVis()
    while True:
        user_input = input(env.time_step + \
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
        decision = env.sdv.get_sdv_decision(env, obs)

        # if env.time_step % 10 == 0:
        #     np.random.seed(None)
        #     decision = np.random.choice([5, 2])
        # print('decision ', decision)
        # print('act_long ', env.sdv.act_long)


        viewer.log_var(env.vehicles)
        viewer.render(env.vehicles, env.sdv)
        env.step(decision)

if __name__=='__main__':
    main()
