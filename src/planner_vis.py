# from agents import Vehicle, EgoVehicle
import os
import sys
# sys.path.insert(0, './src')
import matplotlib.pyplot as plt

from environment.env import Env
import numpy as np
from vehicles.sd_vehicle import SDVehicle

# from graphics import Viewer
# Highway_scene()
import matplotlib.pyplot as plt
# %%
# %%
from vehicles.idm_vehicle import IDMVehicle
env = Env()
# viewer = Viewer(env)
# viewer.plotlive()

v1 = IDMVehicle(id=1, lane_id=3, x=18, v=10)
v2 = IDMVehicle(id=2, lane_id=3, x=20, v=10)
env.sdv = SDVehicle(id='sdv', lane_id=5, x=35, v=10, env=env)
env.vehicles.append(env.sdv)
env.gen_idm_vehicles(gap_rane=[20, 40], v_range=[10, 15], max_vehicle_count=100)

obs = env.reset()
env.render()

for i in range(500):
    # env.render()
    env.sdv.planner.plan(env, obs) # TODO, get visualisation outputs
    decision, decision_counts = env.sdv.planner.get_decision()
    env.viewer.tree_info = env.sdv.planner.tree_info
    env.viewer.belief_info = env.sdv.planner.belief_info
    env.viewer.decision_counts = decision_counts
    env.render()

    # user_in = input("Do you wanna continue?")
    # if user_in != 'y':
    #     break
    # obs, reward, terminal = env.step()
    env.step()
    #
    # ###############
    print('observation: ', obs)
    print('lane_id: ', env.sdv.lane_id)
    print('terminal: ', terminal)
    print('decision: ', decision)
    # ###############
    #
    #
