namespace garf {

    bool is_admissible_split(uint64_t num_going_left, uint64_t num_going_right) {
        return true;
    }

    template<typename FeatT>
    void evaluate_single_split(const feature_mtx<FeatT> & split_feature_values,
                               const data_indices_vec & data_indices,
                               split_idx_t split_feature, FeatT thresh,
                               split_dir_vec * candidate_split_directions,
                               data_indices_vec * indices_going_left,
                               data_indices_vec * indices_going_right,
                               datapoint_idx_t * const num_going_left,
                               datapoint_idx_t * const num_going_right) {

        datapoint_idx_t num_samples_to_evaluate = split_feature_values.rows();
        if (candidate_split_directions->size() != num_samples_to_evaluate) {
            throw new std::logic_error("candidate_split_directions size doesn't match number of sampels to split on");
        }

        // Keep track of places to 
        datapoint_idx_t left_idx = 0;
        datapoint_idx_t right_idx = 0;

        for (datapoint_idx_t i = 0; i < num_samples_to_evaluate; i++) {
            if (split_feature_values(i, split_feature) <= thresh) {
                candidate_split_directions->coeffRef(i) = LEFT;
                indices_going_left->coeffRef(left_idx) = data_indices(i);
                left_idx++;
            } else {
                candidate_split_directions->coeffRef(i) = RIGHT;
                indices_going_right->coeffRef(right_idx) = data_indices(i);
                right_idx++;
            }
        }
        *num_going_left = left_idx;
        *num_going_right = right_idx;
    }


    template<typename FeatT, typename LabT>
    bool AxisAlignedSplFitter<FeatT, LabT>::choose_split_parameters(const feature_mtx<FeatT> & features,
                                                       const feature_mtx<LabT> & labels,
                                                       const data_indices_vec & parent_data_indices,
                                                       const MultiDimGaussianX<LabT> & parent_dist,
                                                       AxisAlignedSplt<FeatT> * split,
                                                       data_indices_vec * left_child_indices_out,
                                                       data_indices_vec * right_child_indices_out) {
        // This does all the work of the splitting. We take in the dataset,
        // a list of what ended up in the parent data node (which we are currently
        // splitting), and the distribution it caused. We output into the split node,
        // and the vector of child indices.
        datapoint_idx_t num_in_parent = parent_data_indices.size();
        feat_idx_t feature_dimensionality = features.cols();

        const SplitOptions & split_opts = this->split_opts;

        // Pick which features to try from a uniform distribution
        std::uniform_int_distribution<feat_idx_t> feat_idx_dist(0, feature_dimensionality-1);
        feat_idx_vec feature_indices_to_evaluate(split_opts.num_splits_to_try);
        for (feat_idx_t i = 0; i < split_opts.num_splits_to_try; i++) {
            feature_indices_to_evaluate(i) = feat_idx_dist(this->rng);
        }
        std::cout << "parent_data_indices = " << parent_data_indices.transpose()
            << " feat_indices = " << feature_indices_to_evaluate.transpose() << std::endl;

        // Evaluate datapoint at each of these feature indices
        feature_mtx<FeatT> feature_values(num_in_parent, split_opts.num_splits_to_try);
        for (datapoint_idx_t data_idx = 0; data_idx < num_in_parent; data_idx++) {
            for (feat_idx_t feat_idx = 0; feat_idx < split_opts.num_splits_to_try; feat_idx++) {
                feature_values(data_idx, feat_idx) = features(parent_data_indices(data_idx), feature_indices_to_evaluate(feat_idx));
            }
        }
        std::cout << "feature values = " << std::endl << feature_values << std::endl;

        // Work out the min and max range
        feature_vec<FeatT> min_feature_values = feature_values.colwise().minCoeff();
        feature_vec<FeatT> max_feature_values = feature_values.colwise().maxCoeff();
        std::cout << "min features: " << min_feature_values << std::endl;
        std::cout << "max features: " << max_feature_values << std::endl;

        // Generate some feature splits which land inside these points
        feature_mtx<FeatT> split_thresholds(split_opts.num_splits_to_try, split_opts.threshes_per_split);
        for (feat_idx_t feat_idx = 0; feat_idx < split_opts.num_splits_to_try; feat_idx++) {
            // std::cout << feat_idx << ": for feature " << feature_indices_to_evaluate(feat_idx)
            //     << " generating threshes in [" << min_feature_values(feat_idx) << "," << max_feature_values(feat_idx) << "]" << std::endl;

            std::uniform_real_distribution<FeatT> thresh_dist(min_feature_values(feat_idx), max_feature_values(feat_idx));

            for (split_idx_t split_idx = 0; split_idx < split_opts.threshes_per_split; split_idx++) {
                split_thresholds(feat_idx, split_idx) = thresh_dist(this->rng);
            }
        }

        std::cout << "thresholds = " << std::endl << split_thresholds << std::endl;

        // Store best information gain so far in here
        double best_inf_gain = -std::numeric_limits<LabT>::infinity();
        split_dir_vec candidate_split_directions(num_in_parent);
        bool good_split_found = false;

        data_indices_vec samples_going_left(num_in_parent - 1);
        data_indices_vec samples_going_right(num_in_parent - 1);

        datapoint_idx_t num_going_left, num_going_right;

        for (split_idx_t split_idx = 0; split_idx < split_opts.num_splits_to_try; split_idx++) {
            for (split_idx_t thresh_idx = 0; thresh_idx < split_opts.threshes_per_split; thresh_idx++) {
                evaluate_single_split(feature_values, parent_data_indices, split_idx, split_thresholds(split_idx, thresh_idx),
                                      &candidate_split_directions, &samples_going_left, &samples_going_right,
                                      &num_going_left, &num_going_right);
                std::cout << "split #" << split_idx << " thresh #" << thresh_idx << " = " << split_thresholds(split_idx, thresh_idx);
                std::cout << ", candidate split directions = ";
                for (datapoint_idx_t i = 0; i < num_in_parent; i++) {
                    if (candidate_split_directions(i) == LEFT) {
                        std::cout << "L";
                    } else {
                        std::cout << "R";
                    }
                }
                std::cout << " " << num_going_left << " going left: " << samples_going_left.head(num_going_left).transpose() << " : "
                    << num_going_right << " going right: " << samples_going_right.head(num_going_right).transpose() << std::endl;

                // Now work out the information gain. First fit gaussians
                this->left_child_dist.fit_params(labels, samples_going_left, num_going_left);
                this->right_child_dist.fit_params(labels, samples_going_right, num_going_right);
                std::cout << "P" << num_in_parent << parent_dist
                    << " L" << num_going_left << this->left_child_dist
                    << " R" << num_going_right << this->right_child_dist << " ";

                double inf_gain = information_gain(parent_dist, this->left_child_dist, this->right_child_dist,
                                                   num_in_parent, num_going_left, num_going_right);
                std::cout << "igain: " << inf_gain << std::endl << std::endl;

                // FIXME: also test if this is a decent split at this point
                if ((inf_gain > best_inf_gain)
                    && is_admissible_split(num_going_left, num_going_right)) {
                    
                    // Record that we have found a decent split
                    good_split_found = true;
                    best_inf_gain = inf_gain;

                    // Store the parameters into the split node (which is actually part of the
                    // prediction node).
                    split->feat_idx = feature_indices_to_evaluate(split_idx);
                    split->thresh = split_thresholds(split_idx, thresh_idx);

                    // Send the output data (ie which data goes left / right) to node::train()
                    *left_child_indices_out = samples_going_left.head(num_going_left);
                    *right_child_indices_out = samples_going_right.head(num_going_right);

                    std::cout << "found new best split: " << *split << std::endl;
                }
            }
        }

        return good_split_found;
    }
}