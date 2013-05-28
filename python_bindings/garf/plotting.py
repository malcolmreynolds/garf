
import numpy as np
import matplotlib.pyplot as plt

from garf import set_predict_opts



def leaf_heatmap(forest, feature_vector, max_depth=None, num_bins=100):
    """Given some feature vector, predict (down to a certain
    maximum depth if specified) and draw a heatmap of the
    2d results"""

    correct_feat_vec_shape = (1, forest.stats.data_dimensions)

    if feature_vector.shape != correct_feat_vec_shape:
        raise ValueError('feature vector shape is %s,  must be %s' %
                         (feature_vector.shape, correct_feat_vec_shape))

    num_trees = forest.stats.num_trees

    with set_predict_opts(forest, maximum_depth=max_depth):
        mu, var, leaf_indices = forest.predict(example, output_leaf_indices=True)
        leaf_indices = leaf_indices.reshape(num_trees)  # we only hae one row of results

    return heatmap_from_leaf_indices(forest, leaf_indices, num_bins=num_bins)

    # pred_values = np.zeros((2, num_trees))
    # # We only provided one

    # for idx, (l_id, tree) in enumerate(zip(leaf_indices, forest.all_trees())):
    #     pred_values[:, idx] = tree.get_node(l_id).dist.mean.flatten()

    # x_values, y_values = pred_values[0, :], pred_values[1, :]

    # x_min, x_max = x_values.min(), x_values.max()
    # y_min, y_max = y_values.min(), y_values.max()

    # heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=num_bins)


def heatmap_from_leaf_indices(forest, leaf_indices, num_bins=100):
    num_trees = forest.stats.num_trees
    if leaf_indices.shape != (num_trees,):
        raise ValueError("leaf_indices must be (num_trees,) in size")

    pred_values = np.zeros((2, num_trees))
    # We only provided one

    for idx, (l_id, tree) in enumerate(zip(leaf_indices, forest.all_trees())):
        pred_values[:, idx] = tree.get_node(l_id).dist.mean.flatten()

    x_values, y_values = pred_values[0, :], pred_values[1, :]

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=num_bins)
    return heatmap, xedges, yedges


def plot_heatmap(heatmap, xedges, yedges):
    """Actually plot the heatmap"""
    # This is a hacky way whih 
    plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], interpolation='nearest')
