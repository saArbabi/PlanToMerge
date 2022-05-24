import numpy as np
import os
import sys
from graphviz import Digraph
import time
import collections


# sys.path.append('C:\\Users\\sa00443\\.virtualenvs\\PlanToMerge-7mESgO4X\\Lib\\site-packages')
# sys.path.append('C:\\Program Files\\Graphviz\\bin')
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
# sys.executable
# %%
class TreeVis():
    OPTIONS = {1 : ['LANEKEEP', 'UP'],
               2 : ['LANEKEEP', 'IDLE'],
               3 : ['LANEKEEP', 'DOWN'],
               4 : ['MERGE', 'IDLE'],
               5 : ['ABORT', 'IDLE']}

    def __init__(self):
        self.decision_node_names = ['DecisionNode', 'BeliefNode']
        self.chance_node_names = ['ChanceNode', 'SubChanceNode']

    def create_node_label(self, tree_node, node_title, node_type):
        """
        Labels are what is shown on each node.
        """
        count = tree_node.count
        if node_type in self.decision_node_names:
            node_label = str(count)

        elif node_type in self.chance_node_names:
            value = round(tree_node.value, 2)
            node_label = f'{node_title}\n count; {count}\n value; {value}'

        return node_label

    def add_node_to_graph(self, child_node, most_visited, node_type, penwidth):
        if node_type in self.decision_node_names:
            self.graph.node(child_node.node_name, shape='circle', label=child_node.node_label, penwidth=penwidth)

        elif node_type in self.chance_node_names:
            if most_visited:
                self.graph.node(child_node.node_name, style='filled',
                       fillcolor='lightgreen',
                       label=child_node.node_label,
                       penwidth=penwidth)
            else:
                self.graph.node(child_node.node_name, label=child_node.node_label, penwidth=penwidth)

    def add_root_to_graph(self, parent_node):
        penwidth = self.get_penwidth(parent_node)
        node_name = str(id(parent_node))
        parent_node.node_label = self.create_node_label(parent_node, 'root', 'DecisionNode')
        parent_node.node_name = str(id(parent_node))
        self.graph.node(node_name, color='red', shape='circle', label=parent_node.node_label, penwidth=penwidth)

    def add_branch_to_graph(self, parent_node):
        """ The parent must already have been added to the tree
        (with its given node_label and node_name). Here its children are added.
        """
        counts = [child_node.count for child_node in parent_node.children.values()]
        max_count = max(counts)
        parent_node.children = collections.OrderedDict(sorted(parent_node.children.items()))

        for key, child_node in parent_node.children.items():
            try:
                node_title = self.OPTIONS[key][0]+'_'+self.OPTIONS[key][1]
            except:
                node_title = None

            node_type = type(child_node).__name__
            child_node.node_label = self.create_node_label(child_node, node_title, node_type)
            child_node.node_name = str(id(child_node))

            if child_node.count == max_count:
                most_visited = True
            else:
                most_visited = False

            penwidth = self.get_penwidth(child_node)
            self.add_node_to_graph(child_node, most_visited, node_type, penwidth)
            self.graph.edge(parent_node.node_name, child_node.node_name, penwidth=penwidth)

    def grab_child_nodes(self, nodes):
        """
        Return the child node of all the nodes in the input list.
        Note: All the children are of the same type and are at the same depth within
        the tree.
        """
        next_depth_level_nodes = []
        for node in nodes:
            next_depth_level_nodes.extend(list(node.children.values()))
        return next_depth_level_nodes

    def get_penwidth(self, child_node):
        """
        The more time a node is visited, the thicker its penwidth
        """
        return str(max([1, int(child_node.count*8/self.root_count)]))

    def construct_graphviz(self, tree_obj):
        node = tree_obj.root
        self.add_root_to_graph(node)
        self.add_branch_to_graph(node)
        next_depth_level_nodes = [node]
        while True:
        # for i in range(5):
            next_depth_level_nodes = self.grab_child_nodes(next_depth_level_nodes)
            if not next_depth_level_nodes:
                break
            for node in next_depth_level_nodes:
                if node.children:
                    self.add_branch_to_graph(node)

    def save_tree_snapshot(self, tree_obj, time_step):
        """
        param tree_obj: Instance of the search tree object
        Save format: pdf
        """
        timestr = time.strftime("%Y%m%d-%H-%M-%S")
        tree_name = f'{timestr}-search_tree_step_{time_step}'
        self.graph = Digraph(tree_name, node_attr={'shape': 'rectangle', 'fontsize':'11'})
        self.root_count = tree_obj.root.count
        self.construct_graphviz(tree_obj)
        self.graph.render(directory='tree_snapshots')
        print('SAVE TREE ' + tree_name)
