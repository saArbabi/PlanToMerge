import sys
sys.path.insert(0, './src')
from plan_viewer import Viewer
from envs.auto_merge import EnvAutoMerge
import numpy as np
import json
from tree_search.factory import safe_deepcopy_env
import matplotlib.pyplot as plt


def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvAutoMerge(config)
    episode_id = 3
    env.initialize_env(episode_id)

    viewer = Viewer(config)

    decisions = [4, 1, 4, 1, 4]
    decision_i = 0
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


        obs = env.planner_observe()
        ###############################################
        viewer.draw_highway(viewer.env_ax, env.vehicles)

        ###############################################
        if env.sdv.time_lapse %  env.sdv.decision_steps_n == 0:
            env.sdv.planner.reset()
            for i in range(1, env.sdv.planner.config['budget']+1):
                print('MCTS Run ', i, '/', env.sdv.planner.config['budget'])
                env.sdv.planner.run(safe_deepcopy_env(env), obs)
                _, env.sdv.decisions_and_counts = env.sdv.planner.get_decision()
                viewer.draw_decision_counts(viewer.decision_ax, env.sdv)
                viewer.draw_plans(viewer.env_ax, env.sdv)
                viewer.draw_beliefs(viewer.env_ax, env.sdv)
                viewer.fig.tight_layout()
                plt.pause(1e-10)

                ###############################################
                user_input = input()
                if user_input:
                    if user_input == 'n':
                        sys.exit()

            decision, env.sdv.decisions_and_counts = env.sdv.planner.get_decision()

            viewer.fig.tight_layout()
            plt.pause(1e-10)

        else:
            decision = env.sdv.decision
        ##############################################

        # decision = env.sdv.get_sdv_decision(env, obs)

        # if env.time_step % 10 == 0:
        #     decision = decisions[decision_i]
        #     decision_i += 1
        #



        viewer.render(env.vehicles, env.sdv)
        env.step(decision)

if __name__=='__main__':
    main()
