namespace garf {

    template<typename FeatT, typename LabT>
    bool SplFitter<FeatT, LabT>::is_admissible_split(eigen_idx_t num_going_left, eigen_idx_t num_going_right) const {
        if ((num_going_right < split_opts.num_per_side_for_viable_split) ||
            (num_going_left < split_opts.num_per_side_for_viable_split)) {
            return false;
        }
        return true;
    }

    template<typename FeatT, typename LabT>
    void SplFitter<FeatT, LabT>::generate_split_thresholds() {
        const feat_idx_t num_splits_to_try = split_opts.num_splits_to_try;
        const split_idx_t threshes_per_split = split_opts.threshes_per_split;

        for (feat_idx_t feat_idx = 0; feat_idx < num_splits_to_try; feat_idx++) {
            std::uniform_real_distribution<FeatT> thresh_dist(min_feature_values(feat_idx), max_feature_values(feat_idx));

            for (split_idx_t split_idx = 0; split_idx < threshes_per_split; split_idx++) {
                split_thresholds(feat_idx, split_idx) = thresh_dist(rng);
            }
        }
    }


    template<typename FeatT, typename LabT>
    void SplFitter<FeatT, LabT>::check_split_thresholds() {
        const feat_idx_t num_splits_to_try = split_opts.num_splits_to_try;
        const split_idx_t threshes_per_split = split_opts.threshes_per_split;

        for (feat_idx_t feat_idx = 0; feat_idx < num_splits_to_try; feat_idx++) {
            for (split_idx_t split_idx = 0; split_idx < threshes_per_split; split_idx++){
                FeatT thresh = split_thresholds(feat_idx, split_idx);
                if (thresh < min_feature_values(feat_idx) ||
                    thresh > max_feature_values(feat_idx)) {
                    std::cout << "feature idx " << feat_idx << " split idx " << split_idx << " = " << thresh 
                        << " which is outside range [" << min_feature_values(feat_idx)
                        << ", " << max_feature_values(feat_idx) << ")" << std::endl;
                }
            }
        }
    }


    template<typename FeatT, typename LabT>
    void SplFitter<FeatT, LabT>::find_min_max_features(const datapoint_idx_t num_in_parent) {
        // The feature_values matrix is almost certainly bigger than we need, so only look at the top rows
        // when working out the min and max
        min_feature_values = candidate_feature_values.topRows(num_in_parent).colwise().minCoeff();
        max_feature_values = candidate_feature_values.topRows(num_in_parent).colwise().maxCoeff();
    }

    template<typename FeatT, typename LabT>
    void SplFitter<FeatT, LabT>::evaluate_single_split(const data_indices_vec & data_indices,
                                                                  const datapoint_idx_t num_in_parent,
                                                                  split_idx_t split_feature, FeatT thresh,
                                                                  split_dir_vec * candidate_split_directions,
                                                                  data_indices_vec * const indices_going_left,
                                                                  data_indices_vec * const indices_going_right,
                                                                  datapoint_idx_t * const num_going_left,
                                                                  datapoint_idx_t * const num_going_right) const {
        // Keep track of places to 
        datapoint_idx_t left_idx = 0;
        datapoint_idx_t right_idx = 0;

        for (datapoint_idx_t i = 0; i < num_in_parent; i++) {
            if (this->candidate_feature_values(i, split_feature) <= thresh) {
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


}