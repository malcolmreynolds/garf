namespace garf {

    template<typename FeatT, typename LabT>
    void AxisAlignedSplFitter<FeatT, LabT>::select_candidate_features() {
        const feat_idx_t num_splits_to_try = this->split_opts.num_splits_to_try;
        for (feat_idx_t i = 0; i < num_splits_to_try; i++) {
            feature_indices_to_evaluate(i) = feat_idx_dist(this->rng);
        }        
    }

    // Fill in the top most `num_in_parent` rows of the feature_values matrix with the selected
    // features from our overall feature matrices
    template<typename FeatT, typename LabT>
    void AxisAlignedSplFitter<FeatT, LabT>::evaluate_datapoints_at_each_feature(const feature_mtx<FeatT> & features,
                                                                                const data_indices_vec & parent_data_indices,
                                                                                const datapoint_idx_t num_in_parent) {
        const feat_idx_t num_splits_to_try = this->split_opts.num_splits_to_try;

        for (datapoint_idx_t data_idx = 0; data_idx < num_in_parent; data_idx++) {
            for (feat_idx_t feat_idx = 0; feat_idx < num_splits_to_try; feat_idx++) {
                this->candidate_feature_values(data_idx, feat_idx) = features(parent_data_indices(data_idx),
                                                                              feature_indices_to_evaluate(feat_idx));
            }
        }
    }

    template<typename FeatT, typename LabT>
    void AxisAlignedSplFitter<FeatT, LabT>::set_parameters_in_splitter(const split_idx_t split_idx,
                                                                       const split_idx_t thresh_idx,
                                                                       AxisAlignedSplt<FeatT> * const splitter) {
        splitter->feat_idx = this->feature_indices_to_evaluate(split_idx);
        splitter->thresh = this->split_thresholds(split_idx, thresh_idx);
    }


    template<typename FeatT, typename LabT>
    bool AxisAlignedSplFitter<FeatT, LabT>::choose_split_parameters(const feature_mtx<FeatT> & all_features,
                                                       const feature_mtx<LabT> & all_labels,
                                                       const data_indices_vec & parent_data_indices,
                                                       const util::MultiDimGaussianX<LabT> & parent_dist,
                                                       AxisAlignedSplt<FeatT> * const split,
                                                       data_indices_vec * left_child_indices_out,
                                                       data_indices_vec * right_child_indices_out) {
        const datapoint_idx_t num_in_parent = parent_data_indices.size();
        const SplitOptions & split_opts = this->split_opts;
#ifdef VERBOSE
        std::cout << "candidate_feature_values.shape = " << candidate_feature_values.rows()
            << "x" << candidate_feature_values.cols() << std::endl;
#endif

        select_candidate_features();
#ifdef VERBOSE
        std::cout << "parent_data_indices = " << parent_data_indices.transpose()
            << " num_in_parent = " << num_in_parent 
            << " feat_indices = " << feature_indices_to_evaluate.transpose() << std::endl;
#endif

        evaluate_datapoints_at_each_feature(all_features, parent_data_indices, num_in_parent);
#ifdef VERBOSE
        std::cout << "feature values = " << candidate_feature_values.topRows(num_in_parent) << std::endl;
        // std::cout << "full feature values = " << candidate_feature_values << std::endl;
#endif

        this->find_min_max_features(num_in_parent);
#ifdef VERBOSE
        std::cout << "min features: " << min_feature_values << std::endl;
        std::cout << "max features: " << max_feature_values << std::endl;
#endif

        this->generate_split_thresholds();
#ifdef VERBOSE
        std::cout << "thresholds = " << std::endl << split_thresholds << std::endl;
#endif

        // this->check_split_thresholds();

        // Store best information gain so far in here
        this->best_inf_gain = -std::numeric_limits<LabT>::infinity();
        this->good_split_found = false;
        // Create references to access parent class public variables. These normally require
        // the use of this-> because of some template bullshit - thanks C++
        datapoint_idx_t & num_going_left = this->num_going_left;
        datapoint_idx_t & num_going_right = this->num_going_right;

        num_going_left = num_going_right = 0;

        // return evaluate_quality_of_each_split(parent_data_indices, num_in_parent,
        //                                       left_child_indices_out, right_child_indices_out);

        for (split_idx_t split_idx = 0; split_idx < split_opts.num_splits_to_try; split_idx++) {
            for (split_idx_t thresh_idx = 0; thresh_idx < split_opts.threshes_per_split; thresh_idx++) {
                this->evaluate_single_split(parent_data_indices, num_in_parent, split_idx, this->split_thresholds(split_idx, thresh_idx),
                                            &this->candidate_split_directions, &this->samples_going_left, &this->samples_going_right,
                                            &num_going_left, &num_going_right);
#ifdef VERBOSE                
                std::cout << "split #" << split_idx << " thresh #" << thresh_idx << " = " << split_thresholds(split_idx, thresh_idx);
                std::cout << ", feature range is [" << min_feature_values(split_idx) << "," << max_feature_values(split_idx);
                std::cout << "], candidate split directions = ";
                for (datapoint_idx_t i = 0; i < num_in_parent; i++) {
                    if (this->candidate_split_directions(i) == LEFT) {
                        std::cout << "L";
                    } else {
                        std::cout << "R";
                    }
                }
                std::cout << " " << num_going_left << " going left: [" << samples_going_left.head(num_going_left).transpose() << "] : "
                    << num_going_right << " going right: [" << samples_going_right.head(num_going_right).transpose() << "]" << std::endl;
#endif
                if ((num_going_left == 0) || (num_going_right == 0)) {

                    // check_split_thresholds();

                    std::cout << "feature values = " << this->candidate_feature_values.topRows(num_in_parent) << std::endl;

                    for (datapoint_idx_t i = 0; i < num_in_parent; i++) {
                        std::cout << i << ":" << parent_data_indices(i) << ": "
                            << all_features.row(parent_data_indices(i)) << std::endl;
                    }


                    // this is a bit weird - figure out why it happened
                    std::cout << "split #" << split_idx << " thresh #" << thresh_idx << " = " << this->split_thresholds(split_idx, thresh_idx);
                    std::cout << ", feature range is [" << this->min_feature_values(split_idx) << "," << this->max_feature_values(split_idx);
                    std::cout << "], candidate split directions = ";
                    for (datapoint_idx_t i = 0; i < num_in_parent; i++) {
                        if (this->candidate_split_directions(i) == LEFT) {
                            std::cout << "L";
                        } else {
                            std::cout << "R";
                        }
                    }
                    std::cout << " " << num_going_left << " going left: [" << this->samples_going_left.head(num_going_left).transpose() << "] : "
                        << num_going_right << " going right: [" << this->samples_going_right.head(num_going_right).transpose() << "]" << std::endl;

                }

                // Now work out the information gain. First fit gaussians
                this->left_child_dist.fit_params(all_labels, this->samples_going_left, num_going_left);
                this->right_child_dist.fit_params(all_labels, this->samples_going_right, num_going_right);
#ifdef VERBOSE
                std::cout << "P" << num_in_parent << parent_dist
                    << " L" << num_going_left << this->left_child_dist
                    << " R" << num_going_right << this->right_child_dist << " ";
#endif

                double inf_gain = information_gain(parent_dist, this->left_child_dist, this->right_child_dist,
                                                   num_in_parent, num_going_left, num_going_right);
#ifdef VERBOSE
                std::cout << "igain: " << inf_gain << std::endl << std::endl;
#endif

                // FIXME: also test if this is a decent split at this point
                if ((inf_gain > this->best_inf_gain)
                    && this->is_admissible_split(num_going_left, num_going_right)) {
                    
                    // Record that we have found a decent split
                    this->good_split_found = true;
                    this->best_inf_gain = inf_gain;

                    // Store the parameters into the split node (which is actually part of the
                    // prediction node).
                    set_parameters_in_splitter(split_idx, thresh_idx, split);

                    // Send the output data (ie which data goes left / right) to node::train()
                    *left_child_indices_out = this->samples_going_left.head(num_going_left);
                    *right_child_indices_out = this->samples_going_right.head(num_going_right);
#ifdef VERBOSE
                    std::cout << "found new best split: " << *split << " - "
                        << num_going_left << "/" << num_going_right << std::endl;
#endif
                }
            }
        }

        return this->good_split_found;
    }
}