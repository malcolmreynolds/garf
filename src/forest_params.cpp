#include "forest_params.hpp"

#include <iostream>

namespace garf {
	
	forest_params::forest_params() {
        std::cout << "constructing forest_params" << std::endl;
		_max_num_trees = 1;
		_max_tree_depth = 1;
		_min_sample_count = 2;
		_num_splits_to_try = 10;
        _num_threshes_per_split = 30;
        _split_type = AXIS_ALIGNED;
        _affinity_distance_type = GAUSSIAN;
        _bagging = false;
        _min_variance = 0.0; // by default we don't do termination based on this
        _balance_bias = 0.1; // by default, only look at entropy
	}
	
    // These may come in handy at some point - either way, little overhead to have them here
	manifold_forest_params::manifold_forest_params() { }
	regression_forest_params::regression_forest_params() { }
	
}
	
