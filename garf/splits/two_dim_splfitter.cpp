namespace garf {


    template<typename FeatT, typename LabT>
    void TwoDimSplFitter<FeatT, LabT>::select_candidate_features() {
        const feat_idx_t num_splits_to_try = this->split_opts.num_splits_to_try;
        for (feat_idx_t i = 0; i < num_splits_to_try; i++) {
            // Sample some feature indices
            feat_indices_1_to_evaluate(i) = feat_idx_dist(this->rng);
            feat_indices_2_to_evaluate(i) = feat_idx_dist(this->rng);

            // Sample some weights
            weights_1_to_evaluate(i) = weight_dist(this->rng);
            weights_2_to_evaluate(i) = weight_dist(this->rng);
        }        
    }

    template<typename FeatT, typename LabT>
    void TwoDimSplFitter<FeatT, LabT>::evaluate_datapoints_at_each_feature(const feature_mtx<FeatT> & features,
                                             const data_indices_vec & parent_data_indices,
                                             const datapoint_idx_t num_in_parent) {

        const feat_idx_t num_splits_to_try = this->split_opts.num_splits_to_try;

        for (datapoint_idx_t d_id = 0; d_id < num_in_parent; d_id++) {
            for (feat_idx_t f_id = 0; f_id < num_splits_to_try; f_id++) {
                datapoint_idx_t this_datapoint = parent_data_indices(d_id);

                // get the two elements out of the relevant row of the feature vector, then multiply by
                // the individual weights
                double feat_val = weights_1_to_evaluate(f_id) * features(this_datapoint, feat_indices_1_to_evaluate(f_id));
                feat_val += weights_2_to_evaluate(f_id) * features(this_datapoint, feat_indices_2_to_evaluate(f_id));

                // Store in the matrix so we can threshold it, etc.
                this->candidate_feature_values(d_id, f_id) = feat_val;
            }
        }
    }

    // Once we have found a decent split, save the values into the splitter object which will
    // persist after the TwoDimSplFitter is destroye.d
    template<typename FeatT, typename LabT>
    void TwoDimSplFitter<FeatT, LabT>::set_parameters_in_splitter(const split_idx_t split_idx,
                                                                  const split_idx_t thresh_idx,
                                                                  TwoDimSplt<FeatT> * const splitter) {
        splitter->feat_1 = this->feat_indices_1_to_evaluate(split_idx);
        splitter->feat_2 = this->feat_indices_2_to_evaluate(split_idx);

        splitter->weight_feat_1 = this->weights_1_to_evaluate(split_idx);
        splitter->weight_feat_2 = this->weights_2_to_evaluate(split_idx);

        splitter->thresh = this->split_thresholds(split_idx, thresh_idx);
    }

    template<typename FeatT, typename LabT>
    bool TwoDimSplFitter<FeatT, LabT>::choose_split_parameters(const feature_mtx<FeatT> & all_features,
                                                       const feature_mtx<LabT> & all_labels,
                                                       const data_indices_vec & parent_data_indices,
                                                       const util::MultiDimGaussianX<LabT> & parent_dist,
                                                       TwoDimSplt<FeatT> * const split,
                                                       data_indices_vec * left_child_indices_out,
                                                       data_indices_vec * right_child_indices_out) {
        const datapoint_idx_t num_in_parent = parent_data_indices.size();
        const SplitOptions & split_opts = this->split_opts;

        select_candidate_features();
        evaluate_datapoints_at_each_feature(all_features, parent_data_indices, num_in_parent);
        this->find_min_max_features(num_in_parent);
        this->generate_split_thresholds();

        // this->check_split_thresholds();

        // Store best information gain so far in here
        this->best_inf_gain = -std::numeric_limits<LabT>::infinity();
        this->good_split_found = false;
        // Create references to access parent class public variables. These normally require
        // the use of this-> because of some template bullshit - thanks C++
        datapoint_idx_t & num_going_left = this->num_going_left;
        datapoint_idx_t & num_going_right = this->num_going_right;

        num_going_left = num_going_right = 0;

        for (split_idx_t split_idx = 0; split_idx < split_opts.num_splits_to_try; split_idx++) {
            for (split_idx_t thresh_idx = 0; thresh_idx < split_opts.threshes_per_split; thresh_idx++) {
                this->evaluate_single_split(parent_data_indices, num_in_parent, split_idx, this->split_thresholds(split_idx, thresh_idx),
                                            &this->candidate_split_directions, &this->samples_going_left, &this->samples_going_right,
                                            &num_going_left, &num_going_right);

                if ((num_going_left == 0) || (num_going_right == 0)) {

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

                double inf_gain = information_gain(parent_dist, this->left_child_dist, this->right_child_dist,
                                                   num_in_parent, num_going_left, num_going_right);

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
                }
            }
        }

        return this->good_split_found;
    }

}