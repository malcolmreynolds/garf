
from _garf import *

import object_list


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


class node_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = object_list._all_nodes
        self.py_binding_name = py_binding_name




# We can also add print functions to stats, options classes, etc.
# Technically the decorator isn't needed, as there is only one
# (eg) stats class, but this keeps everything looking clean
class stats_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = [ForestStats]
        self.py_binding_name = py_binding_name


class forest_opts_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = [ForestOptions]
        self.py_binding_name = py_binding_name


class tree_opts_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = [TreeOptions]
        self.py_binding_name = py_binding_name


class split_opts_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = [SplitOptions]
        self.py_binding_name = py_binding_name


class predict_opts_func(GarfMultiFuncDecorator):
    def __init__(self, py_binding_name):
        self.obj_list = [PredictOptions]
        self.py_binding_name = py_binding_name
