"""Various crafty python bindings"""

import numpy as np
from datetime import datetime

from _garf import *

import object_list

# As default, use doubles for everything with axis aligned - allows people to do "garf.RegressionForest" and get something useful
RegressionForest = RegressionForest_D_D_AX
RegressionTree = RegressionTree_D_D_AX
RegressionNode = RegressionNode_D_D_AX


# Set the correct types for things. Ideally want an automatic way to do this
RegressionForest_D_D._feat_type = np.float64
RegressionForest_D_D._label_type = np.float64

RegressionForest_F_F._feat_type = np.float32
RegressionForest_F_F._label_type = np.float32


class GarfMultiFuncDecorator(object):
    """Decorator for functions which need to be added to ALL
    of the classes in some category - eg a function which must be available
    to all Forest classes. Due to python not knowing about C++ templates,
    we have multiple forest classes available, and we want the same functions
    available on each. This allows us to write a function intended for (eg)
    a forest only once, with the decorator, and code at the end of garf.py
    will automatically add it to all appropriate classes.

    Unlike normal decorators we aren't modifying the 'wrapped'
    function at all, that is returned as normal!

    The superclass-ness just allows me to write this explanation only
    once, but I have a Forest version and Tree version. could write
    a Node version too (just another subclass) if ever needed.."""
    def __call__(self, f):
        # Add this function as an attribute to all the relevant classes,
        # ie all the forest classes or all the tree classes
        for obj in self.obj_list:
            setattr(obj, self.py_binding_name, f)
        return f


class forest_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = object_list._all_forests
        self.py_binding_name = py_binding_name


class tree_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = object_list._all_trees
        self.py_binding_name = py_binding_name


@forest_func("any_invalid_numbers")
def _any_invalid_numbers(vals):
    """Checks if there any numbers like NaN, Inf, which tend to screw things up"""
    return not np.all(np.isfinite(vals))


# def _setattr_no_extra_attributes_allowed(self, attribute, value):
#     """Assign this to the __setattr__ of an object to prevent any
#     new attributes getting added. This is most useful for the classes
#     which represent forest options, like if you misspell an option
#     by doing forest_options.max_noum_trees instead of max_num_trees,
#     this makes that become an error."""
#     # Normally we would do 'if not attribute in self.__dict__'
#     try:
#         self.__getattribute__(attribute)
#     except AttributeError:
#         print "cannot set %s" % attribute
#     self.
#     if not attribute in self.__dict__:
#         print "Cannot set %s" % attribute
#     else:
#         self.__dict__[attribute] = value


@forest_func("l")
def _log_wrapper(self, *args):
    """Allows objects to print stuff with a cool timestamp"""
    print datetime.now().strftime('%Y%m%d:%H%M:')[2:],
    print str(self) + ": ",
    print ' '.join([str(a) for a in args])


# Fix up the forest training code, as we always seem to need to do this
@forest_func("train")
def _train_wrapper(self, features, labels, debug=True):
    if _any_invalid_numbers(features):
        raise ValueError('training features contain NaN or infinity')
    if _any_invalid_numbers(labels):
        raise ValueError('training labels contain NaN or infinity')

    print "no invalid values detected at training"

    if len(features.shape) != 2:
        raise ValueError("features.shape must == 2")
    if len(labels.shape) != 2:
        raise ValueError("labels.shape must == 2")

    # Do type checking here - Boost.python not handling it for us any more
    if features.dtype != self._feat_type:
        self.l("casting features to", self._feat_type)
        features = features.astype(self._feat_type)

    if labels.dtype != self._label_type:
        self.l("casting labels to", self._label_type)
        labels = labels.astype(self._label_type)

    try:
        self._train(features, labels)
    except Exception as e:
        self.l(e)
        raise


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

    if _any_invalid_numbers(features):
        raise ValueError("training features contain NaN or infinity")
    if _any_invalid_numbers(labels):
        raise ValueError("training labels contain NaN or infinity")

    num_features = self.stats.data_dimensions
    if importance_out is None:
        importance_out = np.zeros((num_features, 1), dtype=self._importance_type)
    else:
        self.check_array(importance_out, (num_features, 1), self._importance_type)
    self._feature_importance(features, labels, importance_out)
    return importance_out.flatten()


__default_max_depth = 10


# Get different trees / nodes from a trained forest
@forest_func("all_trees")
def _all_trees_wrapper(self):
    '''Return an iterator used to look through all the trees in a forest'''
    if not self.is_trained:
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


@tree_func("make_node_cache")
def _make_node_cache_wrapper(self):
    """Allows trees to access specific nodes by ID - builds a lookup dictionary"""
    self.node_lookup = {}
    for node in self.all_nodes():
        self.node_lookup[node.id] = node


@tree_func("get_node")
def _get_node_wrapper(self, node_id):
    """get a node by node id. Builds a lookup cache the
    first time it's been called on a particular tree."""
    try:
        return self.node_lookup[node_id]
    except AttributeError:
        # cache doesn't exist yet, so make it..
        self.make_node_cache()
        return self.node_lookup[node_id]
