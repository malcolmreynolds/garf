
// #include <glog/logging.h>

namespace garf {

    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::train(const feature_matrix & features, const label_matrix & labels) {
        if (is_trained) {
            throw std::invalid_argument("forest is already trained");
        }

        uint32_t num_datapoints = features.rows();
        uint32_t data_dimensions = features.cols();
        uint32_t label_dimensions = labels.cols();

        if (labels.rows() != num_datapoints) {
            throw std::invalid_argument("number of labels doesn't match number of features");
        }

        std::cout << "Forest[" << this << "] got " << num_datapoints << "x "
            << data_dimensions << " dimensional datapoints with "
            << label_dimensions << " dimensional labels" << std::endl;

        forest_stats.label_dimensions = label_dimensions;
        forest_stats.data_dimensions = data_dimensions;
        forest_stats.num_training_datapoints = num_datapoints;

        trees.reset(new RegressionTree<SplitT, SplFitterT>[forest_options.max_num_trees]);
        forest_stats.num_trees = forest_options.max_num_trees;
        std::cout << "created " << forest_stats.num_trees << " trees" << std::endl;

        for (uint32_t tree_idx = 0; tree_idx < forest_options.max_num_trees; tree_idx++) {
            trees[tree_idx].tree_id = tree_idx;

            indices_vector data_indices(num_datapoints);
            if (forest_options.bagging) {
                throw std::logic_error("bagging not supported yet");
            } else {
                // gives us a vector [0, 1, 2, 3, ... num_data_points-1]
                data_indices.setLinSpaced(num_datapoints, 0, num_datapoints - 1);
            }
            trees[tree_idx].train(features, labels, data_indices, tree_options, split_options);
        }

        // We are done, so set the forest as trained
        is_trained = true;
    }

    // Clears everything in the forest, ie forgets all the training
    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::clear() {
        std::cout << "clearing forest of " << forest_stats.num_trees << " trees." << std::endl;
        trees.reset();
        forest_stats.num_trees = 0;
        is_trained = false;
    }

    // Given a single feature vector, send it down each tree in turn and fill the given scoped 
    // array with pointers to which node it lands at in each
    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::predict_single_vector(const feature_vector & feature_vec,
                                                                     boost::scoped_array<RegressionNode<SplitT, SplFitterT> const *> * leaf_nodes_reached) const {
        for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
            (*leaf_nodes_reached)[t] = &trees[t].evaluate(feature_vec, predict_options);
        }
    }

    // Checks dimensions of labels_out matrix. Throws an error if it is not present or wrong shape.
    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::check_label_output_matrix(label_matrix * const labels_out,
                                                                         feat_idx_t num_datapoints_to_predict) const {
        if (labels_out == NULL) {
            throw std::invalid_argument("predict(): label ouput vector must be supplied!");
        } else if (labels_out->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): labels_out->cols() != trained label dimensions");
        } else if (labels_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): labels_out->rows() doesn't match num features to predict");
        }
    }


    // Returns false if the variances_out matrix is not present (ie we shouldn't bother computing variance),
    // true if it is present and the right shape. Throws a descriptive exception if it is present but the wrong shape
    template<class SplitT, class SplFitterT>
    bool RegressionForest<SplitT, SplFitterT>::check_variance_output_matrix(label_matrix * const variances_out,
                                                                            feat_idx_t num_datapoints_to_predict) const {
        if (variances_out == NULL) {
            return false;  // caller of predict() hasn't supplied a vector output, so don't compute variances
        } else if (variances_out->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): variances_out->cols() != trained label dimensions");
        } else if (variances_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): variances_out->rows() doesn't match num features to predict");
        }
        return true; // all conditions satisfied, so we should return variances
    }

    // As above, returns true if the leaf index output matrix is the right shape
    template<class SplitT, class SplFitterT>
    bool RegressionForest<SplitT, SplFitterT>::check_leaf_index_output_matrix(tree_idx_matrix * const leaf_indices_out,
                                                                              feat_idx_t num_datapoints_to_predict) const {
        if (leaf_indices_out == NULL) {
            return false; // we don't need to compute / return leaf indices
        } else if (leaf_indices_out->cols() != forest_stats.num_trees) {
            throw std::invalid_argument("predict(): leaf_indices_out->cols() != num_trees");
        } else if (leaf_indices_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): leaf_indices_out->rows() != num_datapoints_to_predict");
        }
        return true; // all conditions satisfied, okay to return leaf indices
    }

    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::predict(const feature_matrix & features,
                                                       label_matrix * const labels_out,
                                                       label_matrix * const variances_out,
                                                       tree_idx_matrix * const leaf_indices_out) const {
        if (!is_trained) {
            throw std::invalid_argument("cannot predict, forest not trained yet");
        }

        // Check features
        feat_idx_t num_datapoints_to_predict = features.rows();
        if (features.cols() != forest_stats.data_dimensions) {
            throw std::invalid_argument("predict(): feature_matrix.cols() != trained data dimensions");
        }

        // Don't need to get a boolean variable back from the label_output test function - if it's invalid
        // we simply throw an exception and bail out
        check_label_output_matrix(labels_out, num_datapoints_to_predict);
        bool outputting_variances = check_variance_output_matrix(variances_out, num_datapoints_to_predict);
        bool outputting_leaf_indices = check_leaf_index_output_matrix(leaf_indices_out, num_datapoints_to_predict);

        std::cout << "in predict(), tests passed" << std::endl;

        // Clear the outputs as we will sum into them
        labels_out->setZero();
        if (outputting_variances) {
            variances_out->setZero();
        }
        if (outputting_leaf_indices) {
            leaf_indices_out->setZero();
        }

        // scoped array so we are exception safe. This array contains pointers to const RegressionNodes, so
        // we can't change the node in any way
        boost::scoped_array<RegressionNode<SplitT, SplFitterT> const *> leaf_nodes_reached;
        leaf_nodes_reached.reset(new RegressionNode<SplitT, SplFitterT> const *[forest_stats.num_trees]);

        // NB it kind of sucks to test whether we are doing variance & leaf index outputting on every iteration
        // of the for loop. However to do the tests outside the for loop we'd need 4 different for loops doing every combination
        // of variance yes/no, leaf index yes/no - seems like a lot of code repetition which is going to be hellish for 
        // maintenance. For now I will leave it as is, but this is a FIXME in case prediction performance becomes a bottleneck.
        // I have chosen to do it this way as it means we only need to do the actual predictions - working out which leaf
        // node a particular datapoint lands at - the minimum number of times.
        for (feat_idx_t feat_vec_idx = 0; feat_vec_idx < num_datapoints_to_predict; feat_vec_idx++) {
            std::cout << "predicting on datapoint #" << feat_vec_idx << ": " << features.row(feat_vec_idx).transpose() << std::endl;
            // for each datapoint, we want to work out the set of leaf nodes it reaches, 
            // then worry about whether we are calculating variances or whatever else. We fill our scoped_array
            // with pointers to the leaf node reached by each datapoint
            predict_single_vector(features.row(feat_vec_idx), &leaf_nodes_reached);


            // This if test and the one below are suboptimal, see explanation at top of for loop
            if (!outputting_variances) {
                // Calculate mean only - simplest case, using naive method
                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    labels_out->row(feat_vec_idx) += leaf_nodes_reached[t]->dist.mean;
                }
                labels_out->row(feat_vec_idx) /= forest_stats.num_trees;
            } else {

                // Compute mean and variance at the same time. We are using iterative method for calculating each
                // online (this is most numerically stable) - see http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
                label_vector mu_n(forest_stats.label_dimensions);
                label_vector mu_n_minus_1(forest_stats.label_dimensions);  // mean at previous timestep
                mu_n.setZero();
                mu_n_minus_1.setZero();

                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    // Get const reference to the mean at the leaf node, just so we don't have to do the pointer indirections again
                    const feature_vector & leaf_node_mean = leaf_nodes_reached[t]->dist.mean;

                    // Update the mean
                    mu_n = mu_n_minus_1 + (1.0 / static_cast<double>(t+1)) * (leaf_node_mean - mu_n_minus_1);
                    // sum_x_sq += leaf_node_mean.cWiseProduct(leaf_node_mean);
                    mu_n_minus_1 = mu_n;
                    variances_out->row(feat_vec_idx) += (leaf_node_mean - mu_n_minus_1).cwiseProduct(leaf_node_mean - mu_n);
                }

                // FIXME: swap the two lines below when I have a version of clang++ with the bug fixed
                // labels_out->row(feat_vec_idx) = mu_n;
                labels_out->row(feat_vec_idx).operator=(mu_n);
// 
                // Need this division since the calculation above computes S = num_datapoints * variance.
                // After this division, we just have the variance which is what we want. Let users square root later
                // if they want.
                variances_out->row(feat_vec_idx) /= static_cast<double>(forest_stats.num_trees);
            }

            if (outputting_leaf_indices) {
                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    leaf_indices_out->coeffRef(feat_vec_idx, t) = leaf_nodes_reached[t]->node_id;
                }
            }

            // std::cout << "data point " << features.row(feat_vec_idx) << " landed in nodes: ";
            // for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
            //     std::cout << "[" << leaf_nodes_reached[t]->node_id << ":" << leaf_nodes_reached[t]->dist << "] ";
            // }
            // std::cout << std::endl;
        }
    }
}