#ifndef GARF_MANIFOLD_FOREST_PARAMS_HPP
#define GARF_MANIFOLD_FOREST_PARAMS_HPP

#include <stdint.h>
#include <iostream>

#include "types.hpp"

namespace garf {
    /* Forest stats stores anything that is only known once training data is provided to
       to the forest. This stuff can't be specified except implicitly */
	struct forest_stats {
        tree_idx_t _num_trees;
        feature_idx_t _feature_dimensionality;
        training_set_idx_t _num_training_samples;
    };
    
    /* These are potentially useful in the future if anything forest-type specific needs
       to be added. */
    struct manifold_forest_stats : public forest_stats { };
    struct regression_forest_stats: public forest_stats {
        label_idx_t _label_dimensionality;
    };
	
    /* Forest_params is whatever we specify before training the forest */
	struct forest_params {
		tree_idx_t _max_num_trees; 
		depth_t _max_tree_depth; 
		uint32_t _min_sample_count; // stop splitting at this number of samples
		uint32_t _num_splits_to_try; // number of candidate splits to try
        uint32_t _num_threshes_per_split; // number of random thresholds to test along each splitting dimension
		
        forest_split_type_t _split_type;
        affinity_distances_types_t _affinity_distance_type;
        
        bool _bagging; // whether we resample (with replacement) the dataset that goes into every tree

        double _min_variance; // If the variance at a node is below this then stop splitting

        double _balance_bias;
		
		forest_params();
	};

    // Just in case there are any parameters only needed for manifold forest
    struct manifold_forest_params : public forest_params {
        manifold_forest_params();
    };
    
    // similarly for regression
    struct regression_forest_params : public forest_params {
        regression_forest_params();
    };


}

#endif