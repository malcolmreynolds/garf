from _garf import *
import numpy as np
import pyublas  # this is needed! Do not remove!


# Do a bunch of whacky monkey patching here to allow the C++ objects to get printed.
# I'm sorry in advance for my sins.
def reg_forest_params__str__(self):
    return "Regression Forest Params(max_num_trees=" + str(self.max_num_trees) + \
        ", max_tree_depth=" + str(self.max_tree_depth) + \
        ", min_sample_count=" + str(self.min_sample_count) + \
        ", num_splits_to_try=" + str(self.num_splits_to_try) + \
        ", num_threshes_per_split=" + str(self.num_threshes_per_split) + \
        ", bagging=" + str(self.bagging) + \
        ", balance_bias=" + str(self.balance_bias) + \
        ", min_variance=" + str(self.min_variance) + ")"
RegressionForestParams.__str__ = reg_forest_params__str__


all_forests = [RegressionForestPrwsSingCtF,
               RegressionForestPrwsCtF,
               RegressionForestHyp,
               RegressionForestFltPrwsSingCtF,
               RegressionForestFltPrwsCtF,
               RegressionForestFltHyp,
               RegressionForestFltSmrtHyp,
               RegressionForestFltNcc,
               RegressionForestFltSingDoub,
               RegressionForestFlt]

all_trees = [RegressionTreePrwsSingCtF,
               RegressionTreePrwsCtF,
               RegressionTreeHyp,
               RegressionTreeFltPrwsSingCtF,
               RegressionTreeFltPrwsCtF,
               RegressionTreeFltHyp,
               RegressionTreeFltSmrtHyp,
               RegressionTreeFltNcc,
               RegressionTreeFltSingDoub,
               RegressionTreeFlt]

forests_axis_aligned = [
    RegressionForestFlt,
    RegressionForest
]

trees_axis_aligned = [
    RegressionTreeFlt,
    RegressionTree
]

# The C++ functions for _train and _predict aren't that flexible, so we have
# python wrappers here which add more customizability and then call the
# underlying native function


def any_invalid_numbers(vals):
    '''Determines if any of the values in a thing are NaN or +/- inf'''
    if np.all(np.isfinite(vals)):
        return False
    return True


def resample_labels(x, y, num_bins, debug=False):
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(y, bins=100, label="before")

    if y.shape[1] != 1:
        raise ValueError("resampling only works for 1D data")
    num_datapoints = x.shape[0]
    num_in_each_bin = int(num_datapoints / num_bins)

    # Store indices of the "new" dataset here
    resampled_indices = -1 * np.ones(num_in_each_bin * num_bins, dtype=np.uint32)
    bin_lower_limits = np.linspace(y.min(), y.max(), num_bins + 1)

    # For each bin, get the indices that should be within that bin and then
    # randomly select num_in_each_bin of them
    for i in xrange(num_bins):
        dest_idx = i * num_in_each_bin
        # import pdb; pdb.set_trace()
        labels_in_this_range = ((y > bin_lower_limits[i]) &
                                (y < bin_lower_limits[i + 1])).nonzero()[0]
        chosen_indices = np.random.randint(0, labels_in_this_range.size,
                                           size=num_in_each_bin)
        resampled_indices[dest_idx:(dest_idx + num_in_each_bin)] = \
            labels_in_this_range[chosen_indices]

        # print "filled %d to %d of %d" % (dest_idx, dest_idx + num_in_each_bin, resampled_indices.size)

        if debug:
            print "resampled from ", labels_in_this_range.size, \
                " in this range to pick ", num_in_each_bin

    # import pdb; pdb.set_trace()
    x = x[resampled_indices, :]
    y = y[resampled_indices]

    if debug:
        plt.hist(y, bins=100, label="after")
        plt.title("label resampling")
        plt.legend()
    return x, y


def remove_zero_labels(x, y):
    if y.shape[1] != 1:
        raise ValueError("remove_zeros only works with 1D data")
    indices_to_keep = ((y < -1e-7) | (y > 1e-7)).flatten()

    print("removed zeros, chopped dataset from %d to %d" %
          (x.shape[0], indices_to_keep.sum()))
    return x[indices_to_keep, :].copy(), y[indices_to_keep].copy()


# Fix up the forest training code, as we always seem to need to do this
def train_wrapper(self, features, labels,
                  resample_hist_bins=False, debug=True,
                  remove_zeros=False):
    # print "x.dtype=%s y.dtype=%s" % (x.dtype, y.dtype)

    if any_invalid_numbers(features):
        raise ValueError('training features contain NaN or infinity')
    if any_invalid_numbers(labels):
        raise ValueError('training labels contain NaN or infinity')

    print "no invalid values detected at training"

    # If we are resampling then we need to calculate the range of the labels
    # and then sample evenly from each of these bins
    if resample_hist_bins:
        features, labels = resample_labels(features, labels,
                                           num_bins=resample_hist_bins, debug=debug)
    if remove_zeros:
        features, labels = remove_zero_labels(features, labels)

    # Lots of things can go wrong here and cause boost.python errors, so we try
    # a bunch of ways to fix it
    try:
        self._train(features, labels)
        return
    except TypeError as e:
        pass

    try:
        # Labels might have been passed as a vector but they should be a matrix
        self._train(features, labels.reshape((labels.size, 1)))
        return
    except TypeError as e:
        pass

    # Perhaps the data has been passed as a copy - another way to screw it up
    try:
        self._train(features.copy(), labels.copy())
        return
    except TypeError as e:
        print ("couldn't get training to work: x.dtype = %s y.dtype = %s" %
               (features.dtype, labels.dtype))
        raise


__default_max_depth = (2 ** 31) - 1


def predict_wrapper(self, features,
                    trees_to_predict_with=None,
                    output_leaf_indices=False,
                    use_weighted_average=True,
                    max_depth=__default_max_depth):
    """Given some forest and some features, allocates the correct size
    output matrix for labels (and variance if specified) and returns them.

    Num_trees allows only a subset of the trees to be used, at the moment
    this just supports passing a number and using those trees.

    max_depth has a default which will predict all the way down each tree.
    If desired, can be set to any non-negative integer to limit the predict
    function."""

    if any_invalid_numbers(features):
        raise ValueError('test features contain NaN or +/- inf')
    print "No NANs detected at testing"

    # I know that in python you should never use isinstance... however, if we
    # don't and the user passes something wrong then we get a type exception from boost.python
    # and they are always a pain in the arse to interpret (ie figure out which argument has a type
    # error.) Hence I am removing the opportunity of this one messing things up - if it's not
    # an int we raise an exception
    if not isinstance(max_depth, (int, long)):
        raise ValueError("max_depth must be set to an integer.")
    if max_depth < 0:
        raise ValueError("negative max depth doesn't mean anything")

    # If unsupplied, set to max num
    if trees_to_predict_with is None:
        trees_to_predict_with = self.stats.num_trees

    label_dims = self.stats.label_dimensionality
    num_samples, feature_dimensionality = features.shape
    if feature_dimensionality != self.stats.feature_dimensionality:
        raise ValueError("asked to predict on %d dimensional data but forest was trained on %d dimensions"
            % (feature_dimensionality, self.stats.feature_dimensionality))
    else:
        print "supplied with %d dimensional features which matches forest" % feature_dimensionality

    output_labels = np.zeros((num_samples, label_dims), dtype=np.float32)
    output_var = np.zeros_like(output_labels)

    if output_leaf_indices:
        # import pdb; pdb.set_trace()
        output_leaves = np.zeros((num_samples, trees_to_predict_with), dtype=np.uint32)
        self._predict(features, output_labels, output_var,
                      output_leaves, trees_to_predict_with, max_depth, use_weighted_average)
        self.check_for_negative_variance(output_var)
        return (output_labels, output_var, output_leaves)
    else:
        # import pdb; pdb.set_trace()
        self._predict(features, output_labels, output_var,
                      trees_to_predict_with, max_depth, use_weighted_average)
        self.check_for_negative_variance(output_var)
        return (output_labels, output_var)


def del_wrapper(self):
    print "deleting %s" % self


def check_for_negative_variance_wrapper(self, var):
    print "checking for negative variance..."
    negatives = (var < 0.0)
    if negatives.sum() > 0:
        indices_i, indices_j = negatives.nonzero()
        raise ValueError('Got negative variance from prediction! indices: %s %s' % (indices_i, indices_j))


def all_trees_wrapper(self):
    '''Return an iterator used to look through all the trees in a forest'''
    num_trees = self.stats.num_trees
    for i in xrange(num_trees):
        yield self.get_tree(i)


def tree_feature_frequency_axis_aligned(self):
    '''FIXME: could do this better by adding a method to the splitfinder class
    so that (eg) the hyperplane one could return something different. As it is
    this only works for axis aligned'''
    feat_indices = [n.split.feature for n in self.all_internal()]
    return feat_indices


def forest_feature_frequency_axis_aligned(self):
    features_chosen_per_tree = [t.feature_frequency() for t in self.all_trees()]

    # Need one level of flatten
    features_chosen_per_tree = sum(features_chosen_per_tree, [])

    # Bincount to get the frequency. Make sure we pad it out.
    return np.bincount(features_chosen_per_tree,
                       minlength=self.stats.feature_dimensionality)


# Add our new python functions to the relevant forests (this will happen when
# "import garf" is typed).
for forest in all_forests:
    forest.train = train_wrapper
    forest.predict = predict_wrapper
    forest.all_trees = all_trees_wrapper
    forest.check_for_negative_variance = check_for_negative_variance_wrapper
    forest.__del__ = del_wrapper

for forest in forests_axis_aligned:
    forest.feature_frequency = forest_feature_frequency_axis_aligned


def get_node_wrapper(self, node_idx):
    """Allows us to call tree.get_node(i) in python"""
    try:
        return self.node_cache[node_idx]
    except AttributeError:
        # We haven't generate the index yet.
        self.make_node_cache()
    return self.node_cache[node_idx]


def make_node_cache_wrapper(self):
    """Builds a lookup dict which maps from node indices to nodes"""
    self.node_cache = {}
    for node in self.all_nodes():
        self.node_cache[node.node_id] = node


def all_nodes(self):
    '''Return an iterator to look through all nodes in a tree'''
    nodes_to_visit = [self.root]
    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        yield n
        if not n.is_leaf():
            nodes_to_visit.append(n.l)
            nodes_to_visit.append(n.r)


def all_leaves(self):
    '''Return an iterator through all of tree's leaf nodes'''
    nodes_to_visit = [self.root]
    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        if n.is_leaf():
            yield n
        else:
            nodes_to_visit.append(n.l)
            nodes_to_visit.append(n.r)


def all_internal(self):
    '''Returns an iterator through the tree's internal nodes'''
    nodes_to_visit = [self.root]
    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        if not n.is_leaf():
            nodes_to_visit.append(n.r)
            nodes_to_visit.append(n.l)
            yield n


# Add a function to examine trees
for tree in all_trees:
    tree.make_node_cache = make_node_cache_wrapper
    tree.get_node = get_node_wrapper
    tree.all_nodes = all_nodes
    tree.all_leaves = all_leaves
    tree.all_internal = all_internal

for tree in trees_axis_aligned:
    tree.feature_frequency = tree_feature_frequency_axis_aligned

print "python wrappers added to C++ objects..."


def make_params(options_dict={}, param_type=RegressionForestParams):
    """Returns a parameters object of the desired type, using the
    contents of the dictionary to override any defaults."""
    params = param_type()
    for k, v in options_dict.iteritems():
        if hasattr(params, k):
            params.__setattr__(k, v)
        else:
            raise ValueError("unknown option supplied: %s = %s" % (k, v))
    return params
