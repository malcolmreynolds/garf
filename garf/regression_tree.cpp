#include <glog/logging.h>

namespace garf {

    template<class SplitT>
    void RegressionTree<SplitT>::train(const feature_matrix & features,
                                       const label_matrix & labels,
                                       const indices_vector & data_indices,
                                       const TreeOptions & tree_opts,
                                       const SplitOptions & split_opts) {
        LOG(INFO) << "train() for tree_idx = " << tree_id << std::endl;
    
        // constructor argument to RegressionNode is node id & link to parent,
        // plus label dimensionality (need this in the constructor so
        // we can build our multi dimensional gaussians) and depth.
        // First parameters are zero and NULL since it's the root of the tree.
        root.reset(new RegressionNode<SplitT>(0, NULL, labels.cols(), 0));

        //FIXME! We should also be working out what temporary variables are needed for the training,
        // and generating them all here to pass down the entire tree
        root->train(features, labels, data_indices, tree_opts, split_opts);
    }

    template<class SplitT>
    const RegressionNode<SplitT> & RegressionTree<SplitT>::evaluate(const feature_vector & fvec,
                                                                      const PredictOptions & predict_opts) {
        depth_t current_depth = 0;

        RegressionNode<SplitT> * current_node = root.get();

        split_dir_t dir;
        while (current_depth < predict_opts.maximum_depth) {
            dir = current_node->split.evaluate(fvec);
            if (dir == LEFT) {
                current_node = current_node->left.get();
            } else {
                current_node = current_node->right.get();
            }
            current_depth++;
        }

        return *current_node;

    }
}