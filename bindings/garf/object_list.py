import numpy as np
from _garf import *

"""We need references to all the python objects created by Boost.python
in order to add functions etc to them."""

_all_nodes = [
    RegNode_D_D_AX,
    RegNode_F_F_AX,
    RegNode_D_D_2D,
    RegNode_F_F_2D,

]

_all_forests = [
    RegForest_D_D_AX,
    RegForest_F_F_AX,
    RegForest_D_D_2D,
    RegForest_F_F_2D,

]

_all_trees = [
    RegTree_D_D_AX,
    RegTree_F_F_AX,
    RegTree_D_D_2D,
    RegTree_F_F_2D,
]

_all_nodes = [
    RegNode_D_D_AX,
    RegNode_F_F_AX,
    RegNode_D_D_2D,
    RegNode_F_F_2D,
]

_all_options = [
    ForestOptions,
    TreeOptions,
    SplitOptions,
    PredictOptions
]

_double_feat_forests = [
    RegForest_D_D_2D,
    RegForest_D_D_AX,
]

_double_label_forests = [
    RegForest_D_D_2D,
    RegForest_D_D_AX,
]

_float_feat_forests = [
    RegForest_F_F_2D,
    RegForest_F_F_AX,
]

_float_label_forests = [
    RegForest_F_F_2D,
    RegForest_F_F_AX,
]

for forest in _double_feat_forests:
    forest._feat_type = np.float64

for forest in _double_label_forests:
    forest._label_type = np.float64

for forest in _float_feat_forests:
    forest._feat_type = np.float32

for forest in _float_label_forests:
    forest._label_type = np.float32

for forest in _all_forests:
    # Set the index type we use for return matrices which get passed in.
    # this should be the same for all forests, it's basically the index type
    # (equivalent to eigen_idx_t in the C++ code).
    forest._index_type = np.long
    forest._importance_type = np.float64
