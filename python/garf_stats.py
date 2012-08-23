import numpy as np
"""A bunch of stuff to do stats (but not plotting) of the results of forest predictions"""


def squared_error_vector(predictions, true_labels):
    """For predictions and true_labels being nx3 matrices, returns the
    n element vector of corresponding squared errors."""
    sq_diff = (predictions - true_labels) ** 2  # Still nx3
    return sq_diff.sum(axis=1)  # Now a vector


def mean_squared_error(predictions, true_labels):
    """MSE between predicted and true multi-dimensional label vectors."""
    squared_errors = squared_error_vector(predictions, true_labels)
    return squared_errors.mean()


def average_magnitude(vecs):
    """Get the average magnitude of vectors in the dataset (lets us scale)"""
    magnitudes = np.sqrt((vecs ** 2).sum(axis=1))
    return magnitudes.mean()


def get_leaves(frst_or_tree):
    """Given a forest or a tree, returns a list of the leaf nodes."""
    try:  # Assume it is a tree
        nodes_to_process = [frst_or_tree.root]
    except AttributeError:  # Must actually be a forest
        frst = frst_or_tree
        num_trees = frst.stats.num_trees
        nodes_to_process = [frst_or_tree.get_tree(i).root for i in xrange(num_trees)]

    # Recursively search down tree
    leaves = []
    while nodes_to_process != []:
        n = nodes_to_process.pop()
        if n.is_leaf():
            leaves.append(n)
        else:
            nodes_to_process.append(n.r)
            nodes_to_process.append(n.l)

    return leaves


def get_leaf_depths(frst_or_tree):
    """Given a forest or tree, get a list of the depths of all leaf nodes."""
    leaves = get_leaves(frst_or_tree)
    return [leaf.depth for leaf in leaves]


def get_num_in_each_leaf(frst_or_tree):
    """Given a forest or tree, get a list of the number of samples in each leaf node."""
    leaves = get_leaves(frst_or_tree)
    return [leaf.num_samples for leaf in leaves]


def get_leaf_values(frst_or_tree, dim=0):
    """Given a forest, scans every leaf node which could be used to form a
    prediction and records the value at the given dimension, so that histograms
    etc can be plotted."""

    leaves = get_leaves(frst_or_tree)
    leaf_prediction_values = [leaf.dist.mu[dim] for leaf in leaves]

    return np.array(leaf_prediction_values)


def analyse_results(results):
    """Given the results dict from some run of train_and_test, analyse results"""

    # Work out MSE with no scaling - ideally this would be low
    no_scaling_mse = mean_squared_error(results.pred_y, results.true_y)
    print "MSE with no scaling =", no_scaling_mse

    # prediction_avg_magnitude = average_magnitude(results.pred_y)
    # true_avg_magnitude = average_magnitude(results.true_y)
    # print "true average magnitude =", true_avg_magnitude, \
    #     "predicted average magnitude =", prediction_avg_magnitude

    # scale_factor = true_avg_magnitude / prediction_avg_magnitude
    # scaled_predictions = results.pred_y * scale_factor
    # scaled_predictions_other = results.pred_y / scale_factor

    # scaling_mse = mean_squared_error(scaled_predictions, results.true_y)
    # scaling_other_mse = mean_squared_error(scaled_predictions_other, results.true_y)
    # print "After multiplying vectors by %f : MSE becomes %f (%f%%)" % \
    #     (scale_factor, scaling_mse, 100 * (scaling_mse - no_scaling_mse) / no_scaling_mse)
    # print "dividing vectors by %f : MSE becomes %f (%f%%)" % \
    #     (scale_factor, scaling_other_mse, 100 * (scaling_other_mse - no_scaling_mse) / no_scaling_mse)