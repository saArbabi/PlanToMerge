import numpy as np
import os
import sys
sys.path.append('C:\\Users\\sa00443\\.virtualenvs\\PlanToMerge-7mESgO4X\\Lib\\site-packages')
sys.path.append('C:\\Program Files\\Graphviz\\bin')
sys.path
import graphviz
# os.envpath
os.environ["PATH"]
os.environ["path"]
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
sys.executable
# %%

# %%
import pickle
import sys
sys.path.insert(0, './src')

with open('tree_snap.pickle', 'rb') as handle:
    tree_snap = pickle.load(handle)
# %%

# %%

def create_node_label(tree_node, node_title, node_type):
    """
    Labels are what is shown on each node.
    """
    count = tree_node.count
    if node_type == 'DecisionNode':
        node_label = str(count)

    elif node_type == 'ChanceNode':
        value = round(tree_node.value, 1)
        node_label = f'{node_title}\n count; {count}\n value; {value}'

    return node_label


def add_node_to_graph(child_node, most_visited, node_type, penwidth):
    if node_type == 'DecisionNode':
        w.node(child_node.node_name, shape='circle', label=child_node.node_label, penwidth=penwidth)

    elif node_type == 'ChanceNode':
        if most_visited:
            w.node(child_node.node_name, style='filled',
                   fillcolor='lightgreen',
                   label=child_node.node_label,
                   penwidth=penwidth)
        else:
            w.node(child_node.node_name, label=child_node.node_label, penwidth=penwidth)

def add_root_to_graph(parent_node):
    penwidth = get_penwidth(parent_node)
    node_name = str(id(parent_node))
    parent_node.node_label = create_node_label(parent_node, 'root', 'DecisionNode')
    parent_node.node_name = str(id(parent_node))
    w.node(node_name, color='red', shape='circle', label=parent_node.node_label, penwidth=penwidth)

def add_branch_to_graph(parent_node):
    """ The parent must already have been added to the tree
    (with its gicen node_label and node_name). Here its children are added.
    """
    counts = [child_node.count for child_node in parent_node.children.values()]
    max_count = max(counts)

    for key, child_node in parent_node.children.items():
        if key == 4:
            node_title = 'LC'
        elif key == 1:
            node_title = 'LK'
        else:
            node_title = None


        node_type = type(child_node).__name__
        child_node.node_label = create_node_label(child_node, node_title, node_type)
        child_node.node_name = str(id(child_node))

        if child_node.count == max_count:
            most_visited = True
        else:
            most_visited = False

        penwidth = get_penwidth(child_node)
        add_node_to_graph(child_node, most_visited, node_type, penwidth)
        w.edge(parent_node.node_name, child_node.node_name, penwidth=penwidth)

def grab_child_nodes(nodes):
    """
    Return the child node of all the nodes in the input list.
    Note: All the children are of the same type and are at the same depth within
    the tree.
    """
    next_depth_level_nodes = []
    for node in nodes:
        next_depth_level_nodes.extend(list(node.children.values()))
    return next_depth_level_nodes

def get_penwidth(child_node):
    """
    The more time a node is visited, the thicker its penwidth
    """
    return str(max([1, int(child_node.count*8/20)]))

w = Digraph(node_attr={'shape': 'rectangle', 'fontsize':'11'})
node = tree_snap.root
add_root_to_graph(node)
add_branch_to_graph(node)
next_depth_level_nodes = [node]
while True:
# for i in range(5):
    next_depth_level_nodes = grab_child_nodes(next_depth_level_nodes)
    if not next_depth_level_nodes:
        break
    for node in next_depth_level_nodes:
        if node.children:
            add_branch_to_graph(node)
w
# w.view()
# %%
