#include <glog/logging.h>
#include "regression_forest.hpp"

namespace garf {
    void RegressionTree::train(const feature_matrix & features,
                               const label_matrix & labels,
                               const indices_vector & data_indices,
                               const TreeOptions & tree_opts,
                               const SplitOptions & split_opts) {
        LOG(INFO) << "train() for tree_idx = " << tree_id << std::endl;
    
        // constructor argument to RegressionNode is node id & link to parent,
        // plus label dimensionality (need this in the constructor so
        // we can build our multi dimensional gaussians) and depth.
        // First parameters are zero and NULL since it's the root of the tree.
        root.reset(new RegressionNode(0, NULL, labels.cols(), 0));
        root->train(features, labels, data_indices, tree_opts, split_opts);
    }
}