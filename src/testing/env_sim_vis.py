import sys
sys.path.insert(0, './src')
from vis.viewer import Viewer
from envs.merge import EnvMerge
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvMerge()
    # episode   _id = 504
    # episode_id = 506
    # episode_id = 102
    episode_id = 509
    episode_id = 500
    episode_id = 34
    # episode_id = 10
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    while True:
        if env.time_step >= 0:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            viewer.render(env.vehicles)
            print(env.time_step)

        env.step()

if __name__=='__main__':
    main()
