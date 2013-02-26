// #include <glog/logging.h>

namespace garf {

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionTree<FeatT, LabT, SplitT, SplFitterT>::train(const feature_mtx<FeatT> & features,
                                                                const label_mtx<LabT> & labels,
                                                                const data_indices_vec & data_indices,
                                                                const TreeOptions & tree_opts,
                                                                SplFitterT<FeatT, LabT> * fitter) {
        //LOG(INFO)
#ifdef VERBOSE
        std::cout << "[t" << tree_id << "].train() data_indices = [" << data_indices.transpose() << "]" << std::endl;
#endif


        // std::cout << "[t" << tree_id << "].train() #0, 0: " << features.coeff(0, 0) << " @ " << &features.coeff(0, 0) << std::endl;
        // std::cout << "#0, 0: " << features.coeff(0, 0) << " @ " << &features.coeff(0, 0) << std::endl;

    
        // constructor argument to RegressionNode is node id & link to parent,
        // plus label dimensionality (need this in the constructor so
        // we can build our multi dimensional gaussians) and depth.
        // First parameters are zero and NULL since it's the root of the tree.
        root.reset(new RegressionNode<FeatT, LabT, SplitT, SplFitterT>(0, NULL, labels.cols(), 0));

        // fitter contains whatever temporary variables are needed to fit the object
        // of type SplitT that we are using. This is primarily so that all temporary
        // memory allocation takes place once per tree, rather than at each node, and 
        // then gets automatically deleted because it's on the stack. Also means once training
        // is done only the necessary data is left in the forest (to reduce memory usage
        // & serialization size)
        root->train(*this, features, labels, data_indices,
                    tree_opts, fitter);
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    const RegressionNode<FeatT, LabT, SplitT, SplFitterT> & RegressionTree<FeatT, LabT, SplitT, SplFitterT>::evaluate(const feature_vec<FeatT> & fvec,
                                                                                                                      const PredictOptions & predict_opts) const {
        depth_idx_t current_depth = 0;
#ifdef VERBOSE
        std::cout << "[t" << tree_id << "].predict([" << fvec.transpose()
            << "]), max depth is " << predict_opts.maximum_depth << std::endl;
#endif

        RegressionNode<FeatT, LabT, SplitT, SplFitterT> * current_node = root.get();

        split_dir_t dir;
        while ((current_depth < predict_opts.maximum_depth) &&
               !current_node->is_leaf) {
#ifdef VERBOSE
            std::cout << "t[" << tree_id << ":" << current_node->node_id << "], evaluating..." << std::endl;
#endif
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

    // for a single tree, perform the prediction for a bunch of features and get MSE by
    // comparing to the provided ground truth. We must take in a pointer to predicted_labels_tmp
    // as allocating that internally every time is a big waste.
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    error_t RegressionTree<FeatT, LabT, SplitT, SplFitterT>::test_error(const feature_mtx<FeatT> & features,
                                                                        const label_mtx<LabT> & ground_truth_labels,
                                                                        label_mtx<LabT> * predicted_labels_tmp,
                                                                        const PredictOptions & predict_opts,
                                                                        const datapoint_idx_t num_valid_samples) const {
        for (datapoint_idx_t i = 0; i < num_valid_samples; i++) {
            predicted_labels_tmp->row(i) = evaluate(features.row(i), predict_opts).dist.mean;
        }

        return util::mean_squared_error<LabT>(ground_truth_labels.topRows(num_valid_samples),
                                              predicted_labels_tmp->topRows(num_valid_samples));
    }
}

