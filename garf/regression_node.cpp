// #include <glog/logging.h>

namespace garf {

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionNode<FeatT, LabT, SplitT, SplFitterT>::train(const RegressionTree<FeatT, LabT, SplitT, SplFitterT> & tree,
                                                                const feature_mtx<FeatT> & features,
                                                                const label_mtx<LabT> & labels,
                                                                const data_indices_vec & data_indices,
                                                                const TreeOptions & tree_opts,
                                                                SplFitterT<FeatT, LabT> * fitter,
                                                                const util::MultiDimGaussianX<LabT> * const _dist) {
        // Store the indices which pass through this node - this should do a copy. I hope!
        training_data_indices = data_indices;
        //LOG(INFO)
#ifdef VERBOSE
        std::cout << "[t" << tree.tree_id << ":" << node_id << "] got " << num_training_datapoints()
            << " datapoints: [" << data_indices.transpose() << "]" << std::endl;
#endif
        if (_dist == NULL) {
            //LOG(INFO)
#ifdef VERBOSE
            std::cout << "[t" << tree.tree_id << ":" << node_id << "] no dist provided, calculating..." << std::endl;
#endif
            dist.fit_params(labels, data_indices);
        }
        else {
            //LOG(INFO)
#ifdef VERBOSE
            std::cout << "[t" << tree.tree_id << ":" << node_id << "] using provided distribution" << std::endl;
#endif
            dist.mean = _dist->mean;
            dist.cov = _dist->cov;
        }
        //LOG(INFO)
#ifdef VERBOSE
        std::cout << "[t" << tree.tree_id << ":" << node_id << "] dist = " << dist << std::endl;
#endif

        // std::cout << "[t" << tree.tree_id << ":" << node_id << "] #0, 0: " << features.coeff(0, 0) << " @ " << &features.coeff(0, 0) << std::endl;

        // Check whether to stop growing now. NB: even if this returns false, we might
        // still stop growing if we cannot find a decent split (see below)
        if (stopping_conditions_reached(tree_opts)) {
            return;
        }

        // get the indices going left and right from splitter object. We declare
        // them on the stack here, so that they are cleaned up at the end of this call to train()
        // automatically
        data_indices_vec right_child_indices;
        data_indices_vec left_child_indices;

        // bool good_split_found = true;
        // std::cout << "[t" << tree.tree_id << ":" << node_id << "] choose_split_parameters" << std::endl;
        bool good_split_found = fitter->choose_split_parameters(features, labels, data_indices, dist,
                                                                &split, &left_child_indices, &right_child_indices);

        if (!good_split_found) {
            //LOG(ERROR)
#ifdef VERBOSE_
            std::cout << "[t" << tree.tree_id << ":" << node_id << "] didn't find a good split, stopping" << std::endl;
#endif
            return;
        }

        is_leaf = false;

        // If we are here then assume we found decent splits, indices of which
        // are stored in left_child_indices and right_child_indices. First create child nodes, then
        // do the training. FIXME: we could increase efficiency (slightly!) but
        left.reset(new RegressionNode<FeatT, LabT, SplitT, SplFitterT>(left_child_index(), this,
                                                          labels.cols(), depth + 1));
        right.reset(new RegressionNode<FeatT, LabT, SplitT, SplFitterT>(right_child_index(), this,
                                                           labels.cols(), depth + 1));
        left->train(tree, features, labels, left_child_indices, tree_opts, fitter);
        right->train(tree, features, labels, right_child_indices, tree_opts, fitter);
    }

    // Determine whether the stop growing the tree at this node.
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    bool RegressionNode<FeatT, LabT, SplitT, SplFitterT>::stopping_conditions_reached(const TreeOptions & tree_opts) const {
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