import graphviz
import numpy as np
import os


# %%
from graphviz import Digraph
w = Digraph(node_attr={'shape': 'rectangle'})
w.edges(('0', str(i)) for i in range(1, 10))
w.edges(('1', str(i)) for i in range(25, 27))
w.edges(('25', str(i)) for i in range(30, 32))
decision = 'LC'
count=20
value=10
name = f'{decision} \n count; {count}\n value; {value}'
w.edge('30', name, penwidth='10')
# doctest_mark_exe()
# dir(w.node)
# print(w.source)
w
"""
save a tree object (simple tree, so it can easily be tested)
load the tree here and turn it into tree.
"""
# %%
import pickle
import sys
sys.path.insert(0, './src')

with open('tree_snap.pickle', 'rb') as handle:
    tree_snap = pickle.load(handle)
# %%
"""

"""
def create_node_tag(tree_node, node_label):
    count = tree_node.count
    if type(tree_node).__name__ == 'DecisionNode':
        node_tag = str(count)
        # node_tag = f'{node_label} \n count; {count}'

    elif type(tree_node).__name__ == 'ChanceNode':
        value = round(tree_node.value, 1)
        node_tag = f'{node_label} \n count; {count}\n value; {value}'

    return node_tag, count

def add_children_to_graph(parent, children):

w = Digraph(node_attr={'shape': 'rectangle', 'fontsize':'11', 'margin':'0'})
node = tree_snap.root


parent_tag, _ = create_node_tag(node, 'root')

children_tags = []
children_counts = []
for key, value in node.children.items():
    node_tag, count = create_node_tag(value, str(key))
    children_tags.append(node_tag)
    children_counts.append(count)

w.node(parent_tag, color='red', shape='circle')
max_count_child_index = children_counts.index(max(children_counts))
for i, child_tag in enumerate(children_tags):
    if i == max_count_child_index:
        w.node(child_tag, style='filled', fillcolor='lightgreen')
    else:
        w.node(child_tag)
    w.edge(parent_tag, child_tag)


for parent_node, parent_tag in zip(node.children.values(), children_tags):
    children_tags = []
    children_counts = []
    for key, value in parent_node.children.items():
        node_tag, count = create_node_tag(value, str(key))
        children_tags.append(node_tag)
        children_counts.append(count)

    max_count_child_index = children_counts.index(max(children_counts))
    for i, child_tag in enumerate(children_tags):
        if i == max_count_child_index:
            w.node(child_tag, style='filled', fillcolor='lightgreen', shape='circle', margin='0')
        else:
            w.node(child_tag)
        w.edge(parent_tag, child_tag)
w
# %%

tree_snap.root.children[4].value
type(tree_snap.root.children[4]).__name__
# %%
i = 0
while True:
    print(i)
    i += 1
    if i == 3:
        break

# %%


def add_node_to_graph(parent_dot, child_dot):

tree_snap.value
tree_snap.count
# %%
from graphviz import Digraph
w = Digraph(node_attr={'shape': 'rectangle'})
decision = 'LC'
root = tree_snap[0]
name = f"{'root'} \n count; {root.count}\n value; {round(root.value, 1)}"
w.node(name)
node1 = tree_snap[1]
name1 = f"{'LK'} \n count; {node1.count}\n value; {round(node1.value, 1)}"
w.node(name1)
w.edge(name, name1, penwidth='1')


node2 = tree_snap[2]
name2 = f"{'LC'} \n count; {node2.count}\n value; {round(node2.value, 1)}"
w.node(name2, style='filled', fillcolor='lightgreen')
w.edge(name, name2, penwidth='1')

w

# %%

class ME():
    def __init__(self):
        self.v = 20

me = ME()
me.v
with open('tree_snap.pickle', 'wb') as handle:
    # pickle.dump(env.sdv.planner, handle)
    pickle.dump(me, handle)
