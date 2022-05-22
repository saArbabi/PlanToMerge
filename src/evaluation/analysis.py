
import pickle
planner_name = 'qmdp'
exp_dir = './src/evaluation/experiments/'+ planner_name
with open(exp_dir+'/mc_collection.pickle', 'rb') as handle:
    mc_collection = pickle.load(handle)
mc_collection

# %%
planner_name = 'mcts'
exp_dir = './src/evaluation/experiments/'+ planner_name
with open(exp_dir+'/mc_collection.pickle', 'rb') as handle:
    mc_collection = pickle.load(handle)
mc_collection

# %%
planner_name = 'mcts'
exp_dir = './src/evaluation/experiments/'+ planner_name
with open(exp_dir+'/mc_collection.pickle', 'rb') as handle:
    mc_collection = pickle.load(handle)
mc_collection
