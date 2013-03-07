import numpy as np
from _garf import *

"""We need references to all the python objects created by Boost.python
in order to add functions etc to them."""

_all_nodes = [
    RegressionNode_D_D_AX,
    RegressionNode_F_F_AX,
    RegressionNode_D_D_2D,
    RegressionNode_F_F_2D,

]

_all_forests = [
    RegressionForest_D_D_AX,
    RegressionForest_F_F_AX,
    RegressionForest_D_D_2D,
    RegressionForest_F_F_2D,

]

_all_trees = [
    RegressionTree_D_D_AX,
    RegressionTree_F_F_AX,
    RegressionTree_D_D_2D,
    RegressionTree_F_F_2D,
]

_all_options = [
    ForestOptions,
    TreeOptions,
    SplitOptions,
    PredictOptions
]

for forest in _all_forests:
    # Set the index type we need to use for return matrices which get passed in.
    # this should be the same for all forests, it's basically the index type
    # (equivalent to eigen_idx_t in the C++ code).
    forest._index_type = np.long
    forest._importance_type = np.float64
