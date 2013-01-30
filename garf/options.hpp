#ifndef GARF_OPTIONS_HPP
#define GARF_OPTIONS_HPP

namespace garf {

    /* Options which work at the top level of training - how
      many trees to make, what data to give them etc */
    struct ForestOptions {
        uint32_t max_num_trees;
        bool bagging;
        ForestOptions() : max_num_trees(10), bagging(false) {}
    };

    /* Options which are needed inside a tree - ie when to stop splitting.
      These are needed regardless of how we are doing the splitting */
    struct TreeOptions {
        uint32_t max_depth;
        uint32_t min_sample_count; // don't bother with a split if
        double min_variance;
        TreeOptions() : min_sample_count(10), min_variance(0.00001) {}
    };

    /* Options for how to split the tree */
    struct SplitOptions {
        uint32_t num_splits_to_try;

    };

    /* Options for how we do the prediction (early stopping at maximum depth for example) */
    struct PredictOptions {
        depth_idx_t maximum_depth;
    };
}

#endif