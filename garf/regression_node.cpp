#include <glog/logging.h>
#include "regression_forest.hpp"

namespace garf {

    template<class SplitT>
    void RegressionNode<SplitT>::train(const feature_matrix & features,
                               const label_matrix & labels,
                               const indices_vector & data_indices,
                               const TreeOptions & tree_opts,
                               const SplitOptions & split_opts,
                               const MultiDimGaussianX * const _dist) {
        // Store the indices which pass through this node
        training_data_indices = data_indices;
        LOG(INFO) << "node id " << node_id << " got " << num_training_datapoints() << " datapoints." << std::endl;

        if (_dist == NULL) {
            LOG(INFO) << "no distribution provided, calculating..." << std::endl;
            dist.fit_params(labels, data_indices);
        }
        else {
            LOG(INFO) << "using provided distribution" << std::endl;
            dist.mean = _dist->mean;
            dist.cov = _dist->cov;
        }

        LOG(INFO) << "node id " << node_id << " has mean " << dist.mean << std::endl;

        // Check whether to stop growing now. NB: even if this returns false, we might
        // still stop growing if we cannot find a decent split (see below)
        if (stopping_conditions_reached(tree_opts)) {
            return;
        }

        // Pick how to do the split here
        // indices_vector 
    }


    // Determine whether the stop growing the tree at this node.
    template<class SplitT>
    bool RegressionNode<SplitT>::stopping_conditions_reached(const TreeOptions & tree_opts) const {
        if (depth == tree_opts.max_depth) {
            return true; // Stop growing because we have reached max depth
        }
        if (depth > tree_opts.max_depth) {
            throw std::logic_error("We should never go over the max depth of a tree!");
        }
        if (num_training_datapoints() < tree_opts.min_sample_count) {
            return true; // Stop growing as there are too few datapoints, ie we are overfitting.
        }
        // Sum just the diagonal elements of the covariance matrix. If the variances
        // (diagonal elements) are low / zero then we know the covariances (off diagonal
        // elements) are also low / zero, due to positive semi definite-ness. Note that
        // this should also take care of the case where we have loads of the same element
        // due to bagging.
        double total_variance = dist.cov.diagonal().sum();
        if (total_variance < tree_opts.min_variance) {
            return true;
        }

        return false;  // If none of the above has happened, we keep growing
    }
}