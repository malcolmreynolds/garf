"""Various crafty python bindings"""

import numpy as np
import time
from datetime import datetime

from _garf import *

from forest_decorators import *

# As default, use doubles for everything with axis aligned - allows people to do "garf.RegForest" and get something useful
RegForest = RegForest_D_D_AX
RegTree = RegTree_D_D_AX
RegNode = RegNode_D_D_AX


@forest_func("any_invalid_numbers")
def _any_invalid_numbers(vals):
    """Checks if there any numbers like NaN, Inf, which tend to screw things up"""
    return not np.all(np.isfinite(vals))


@forest_func("l")
def _log_wrapper(self, *args):
    """Allows objects to print stuff with a cool timestamp"""
    print datetime.now().strftime('%Y%m%d:%H%M:')[2:] + self.__str__(short=True, show_id=True) + ":",

    print ' '.join([str(a) for a in args])


@forest_func("train")
def _train_wrapper(self, features, labels, debug=True, calc_importance=False):
    if _any_invalid_numbers(features):
        raise ValueError('training features contain NaN or infinity')
    if _any_invalid_numbers(labels):
        raise ValueError('training labels contain NaN or infinity')

    if len(features.shape) != 2:
        raise ValueError("features.shape must == 2")
    if len(labels.shape) != 2:
        raise ValueError("labels.shape must == 2")

    self.l("training data appears valid")

    # Do type checking here - Boost.python not handling it for us any more
    if features.dtype != self._feat_type:
        self.l("casting features to", self._feat_type)
        features = features.astype(self._feat_type)

    if labels.dtype != self._label_type:
        self.l("casting labels to", self._label_type)
        labels = labels.astype(self._label_type)

    self.l("starting training..")
    start_time = time.clock()
    try:
        self._train(features, labels)
    except Exception as e:
        self.l(e)
        raise

    elapsed_time = (time.clock() - start_time)
    self.l("training done in %.3fs" % elapsed_time)

    if calc_importance:
        self.l("calculating feature importance")
        start_time = time.clock()
        self.importance_vec = self.feature_importance(features, labels)

        elapsed_time = (time.clock() - start_time)
        self.l("feature importance calculated and cached in %.3fs" % elapsed_time)

        # We return the importance vector, but it is
        return self.importance_vec


@forest_func("set_options")
def _set_options_forest_wrapper(self, option_dict):
    """Given a hierarchical options dict, set all the options in a forest

    ie we should expect:

    option_dict = {
        'split_options': {
            'num_splits_to_try': 50,
            'threshes_per_split': 5,
        },
        'tree_options': {
            'max_depth': 5,
            'min_sample_count': 15,
        },
        'forest_options': {
            'bagging': True,
            'max_num_trees': 10,
        },
    }

    Which will set the corresponding values in the split_options,
    tree_options and forest_options (without modifying what is in
    predict_options).
    """

    d_keys = ['tree_options', 'split_options',
              'forest_options', 'predict_options']
    opt_objs = [self.tree_options, self.split_options,
                self.forest_options, self.predict_options]

    for k in option_dict.keys():
        if k not in d_keys:
            raise ValueError('tried to set top level option group %s' % k)

    for opts, d_key in zip(opt_objs, d_keys):
        try:
            opts.set_options(option_dict[d_key])
        except KeyError:
            # Just means user didn't supply any options for that option group
            pass


@forest_func("check_array")
def _check_array(self, array, correct_shape, correct_dtype=None):
    """Check whether a provided array is of the correct shape &
    size, throws exception if not"""
    if array.shape != correct_shape:
        raise ValueError("array provided should be %s but it is %s"
                         % (correct_shape, array.shape))
    if correct_dtype is None:
        # If the second argument is not provided, this function just checks the shape
        return
    if array.dtype != correct_dtype:
        raise ValueError("array provided should have type %s but actually of type %s"
                         % (correct_dtype, array.dtype))


@forest_func("predict")
def _predict_wrapper(self, features, mean_out=None, var_out=None, leaves_out=None, output_leaf_indices=False):
    if not self.trained:
        raise ValueError("cannot predict, forest is not trained")

    # Check the features, get to correct type
    if _any_invalid_numbers(features):
        raise ValueError('training features contain NaN or infinity')

    num_data = features.shape[0]
    self.check_array(features, (num_data, self.stats.data_dimensions))

    if features.dtype != self._feat_type:
        features = features.astype(self._feat_type)

    num_data = features.shape[0]
    if mean_out is None:
        mean_out = np.zeros((num_data, self.stats.label_dimensions), dtype=self._label_type)
    else:
        self.check_array(mean_out, (num_data, self.stats.label_dimensions), self._label_type)

    if leaves_out is None:
        var_out = np.zeros((num_data, self.stats.label_dimensions), dtype=self._label_type)
    else:
        self.check_array(var_out, (num_data, self.stats.num_trees), self._index_type)

    if output_leaf_indices:
        self.l("predicting with output leaves")
        if leaves_out is None:
            leaves_out = np.zeros((num_data, self.stats.num_trees), dtype=self._index_type)
        else:
            self.check_array(leaves_out, (num_data, self.stats.num_trees), self._index_type)
        self._predict(features, mean_out, var_out, leaves_out)
        return mean_out, var_out, leaves_out
    else:
        self.l("predicting with no output leaves")
        self._predict(features, mean_out, var_out)
        return mean_out, var_out


@forest_func("feature_importance")
def _feature_importance_wrapper(self, features, labels, importance_out=None):
    if not self.trained:
        raise ValueError("cannot calculate importance before forest trained")
    try:
        v = self.importance_vec
        self.l("returning cached importance calculated at training time")
        return v
    except AttributeError:
        self.l("importance not cached, calculating......")

    if _any_invalid_numbers(features):
        raise ValueError("training features contain NaN or infinity")
    if _any_invalid_numbers(labels):
        raise ValueError("training labels contain NaN or infinity")

    num_features = self.stats.data_dimensions
    if importance_out is None:
        importance_out = np.zeros((num_features, 1), dtype=self._importance_type)
    else:
        self.check_array(importance_out, (num_features, 1), self._importance_type)

    start_time = time.clock()
    self._feature_importance(features, labels, importance_out)
    end_time = (time.clock() - start_time)
    self.l("importance computed in %.3fs" % end_time)

    # Cache a copy
    self.importance_vec = importance_out.flatten().copy()
    return importance_out.flatten()


@forest_func("clear")
def _clear_wrapper(self):
    """The only thing the C++ doesn't take care of is deleting the importance
    vector which will no longer be valid after retraining, so we delete it here
    before passing onto the C++ clear function."""
    try:
        del self.importance_vec
    except AttributeError:
        # Doesn't matter, just means we hadn't calculated an importance result
        pass

    # Pass through to the C++ clearup
    self._clear()


__default_max_depth = 10


# Get different trees / nodes from a trained forest
@forest_func("all_trees")
def _all_trees_wrapper(self):
    '''Return an iterator used to look through all the trees in a forest'''
    if not self.trained:
        raise ValueError("forest not trained")
    for i in xrange(self.stats.num_trees):
        yield self.get_tree(i)


@tree_func("all_nodes")
def _all_nodes_wrapper(self):
    """Return an iterator over all nodes in a tree"""
    nodes_to_visit = [self.root]

    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        yield n
        if not n.is_leaf:
            nodes_to_visit.extend([n.l, n.r])


@tree_func("all_internal_nodes")
def _all_internal_nodes_wrapper(self):
    """Return an iterator over all internal nodes in a tree"""
    nodes_to_visit = [self.root]

    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        if not n.is_leaf:
            yield n
            nodes_to_visit.extend([n.l, n.r])


@tree_func("all_leaf_nodes")
def _all_leaf_nodes_wrapper(self):
    """Return an iterator over all leaf nodes in a tree"""
    nodes_to_visit = [self.root]

    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        if n.is_leaf:
            yield n
        else:
            nodes_to_visit.extend([n.l, n.r])


@tree_func("all_nodes_at_depth")
def _all_nodes_at_depth_wrapper(self, depth):
    """Return an iterator over all nodes at a certain depth in the tree"""
    if depth < 0:
        raise ValueError("depth must be >= 0")

    nodes_to_visit = [self.root]
    while nodes_to_visit != []:
        n = nodes_to_visit.pop()
        if n.depth == depth:
            # If we are yielding this node because we are at the right depth,
            # no point in traversing below it because everything below it
            # is (by definition) at a greater depth...
            yield n
        elif not n.is_leaf:
            nodes_to_visit.extend([n.l, n.r])


@tree_func("do_preprocessing")
def _do_preprocessing_wrapper(self):
    """Do the stuff that we should do for a tree before
    querying it about stuff."""
    self.make_node_cache()
    self.make_parent_lookup()


@tree_func("make_node_cache")
def _make_node_cache_wrapper(self):
    """Allows trees to access specific nodes by ID - builds a lookup dictionary"""
    self._node_lookup = {}
    for node in self.all_nodes():
        self._node_lookup[node.id] = node


@tree_func("make_parent_lookup")
def _make_parent_lookup_wrapper(self):
    """builds a dict which points from each non-root node
    to its parent. This, in conjunction with the node_lookup,
    allows us to traverse the tree easily (ish)"""
    self._parents = {}
    nodes_to_visit = [self.root]

    while nodes_to_visit:
        n = nodes_to_visit.pop()
        if n.is_leaf:
            continue
        nodes_to_visit.extend([n.l, n.r])
        parent_id = n.id
        self._parents[n.l.id] = parent_id
        self._parents[n.r.id] = parent_id


@tree_func("get_node")
def _get_node_wrapper(self, node_id):
    """get a node by node id. Builds a lookup cache the
    first time it's been called on a particular tree."""
    try:
        return self._node_lookup[node_id]
    except AttributeError:
        # cache doesn't exist yet, so make it..
        self.do_preprocessing()

        return self._node_lookup[node_id]


@tree_func("get_node_path")
def _get_node_path_wrapper(self, node):
    """Given a 'node', returns a list of all nodes
    from root down to and including 'node'"""

    lst = [node]

    still_ascending = True

    while still_ascending:
        # Get the ID here rather than in the try block. This means
        # if we are passed something that isn't a node, then the attributeError
        # we get here is passed out to the caller of get_node_path, rather
        # than triggering an infinite loop below
        node_id = lst[-1].id
        try:
            parent = self.get_node(self._parents[node_id])
            lst.append(parent)
        except KeyError:
            # node doesn't exist in self.parents, so must be root,
            # therefore we are done.
            still_ascending = False
        except AttributeError:
            # self.parents doesn't exist - lets do the required
            # preprocessing
            self.do_preprocessing()

    return list(reversed(lst))


# Print functions
@forest_func("__str__")
def _frst_str_wrapper(self, short=False, show_id=False):
    classname = str(self.__class__)
    dot_pos = classname.find(".")
    end_quote_pos = classname.rfind("'")

    classname = classname[(dot_pos + 1):end_quote_pos]

    s = "[" + classname
    if show_id:
        s += ":" + str(id(self))
    if short:
        return s + "]"
    if not self.trained:
        return s + ":not trained]"

    s += ":" + str(self.stats)

    return s + "]"


@tree_func("__str__")
def _tree_str_wrapper(self):
    c_name = str(self.__class__)
    short_name = c_name[(c_name.find(".") + 1):c_name.rfind("'")]

    s = "[" + short_name
    s += ":#%d" % self.tree_id

    return s + "]"


@node_func("__str__")
def _node_str_wrapper(self):
    c_name = str(self.__class__)
    short_name = c_name[(c_name.find(".") + 1):c_name.rfind("'")]

    s = "[" + short_name
    s += ":#%d" % self.id

    return s + "]"


@opts_func("set_options")
def _set_options_indiv_wrapper(self, opts_dict):
    """Given a dict of string => value mappings, set the corresponding options"""
    for k, v in opts_dict.items():
        # print "setting %s (currently %s) = %s" % (k, self.__getattribute__(k), v)
        self.__setattr__(k, v)
        # print "done: now = %s" % self.__getattribute__(k)


class set_predict_opts(object):
    """Allows us to set forest prediction options in a with
    block and restore the previous settings afterwards"""
    def __init__(self, forest, maximum_depth, **pred_args):
        self.forest = forest
        self.temporary_maximum_depth = maximum_depth
        if pred_args != {}:
            raise ValueError('no optinos apart from maximum depth supported yet')

    def __enter__(self):
        """Save all the predict options set at the moment so we
        can restore them afterwards"""
        # Save the old value and set the new one
        self.old_maximum_depth = forest.predict_options.maximum_depth
        forest.predict_options.maximum_depth = self.temporary_maximum_depth

    def __exit__(self, type, value, traceback):
        forest.predict_options.maximum_depth = self.old_maximum_depth
