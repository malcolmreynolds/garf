"""Perform diagnostics on a forest"""

import matplotlib.pyplot as plt

# http://stackoverflow.com/questions/36932/whats-the-best-way-to-implement-an-enum-in-python
def enum(**enums):
    return type('Enum', (), enums)
    
directions = enum(LEFT=0, RIGHT=1)
    
def gather_route(tree, data_sample):
    "For some tree, return a vector of directions indicating how the sample went down the tree"
    current_node = tree.root
    dirs = []
    try:
        while True:
            if data_sample in current_node.l.samples:
                dirs.append(directions.LEFT)
                current_node = current_node.l
            else:
                dirs.append(directions.RIGHT)
                current_node = current_node.r
    except RuntimeError as re:
        # This is normal, just means we've got to the bottom of the tree so trying
        # to get to current_node.l or current_node.r crashes
        pass
    return dirs
        
def plot_set(tree, depth, labels):
    "Given a tree, Examine the way data is split off to a certain depth. Plot according to labels"
    all_indices = tree.root.samples
    for data_idx in all_indices:
        route = gather_route(tree, data_idx)
        print route