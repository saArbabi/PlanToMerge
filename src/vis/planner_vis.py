import sys
sys.path.insert(0, './src')
from plan_viewer import Viewer
from envs.merge_agent import EnvMergeAgent
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvMergeAgent(config)
    episode_id = 3
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    while True:
        print(env.time_step)

        user_input = input()
        if user_input:
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = user_input
            except:
                pass


        obs = env.sdv.planner_observe()
        decision = env.sdv.get_sdv_decision(env, obs)
        viewer.render(env.vehicles, env.sdv)
        _, reward, terminal = env.step(decision)
        
if __name__=='__main__':
    main()
