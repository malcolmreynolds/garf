#ifndef GARF_NCC_LEVELS_2468_PAIRWISE_SPLTFNDR_IMPL_HPP
#define GARF_NCC_LEVELS_2468_PAIRWISE_SPLTFNDR_IMPL_HPP

namespace garf {
    
    template<typename FeatT, typename LabelT>
    ncc_levels_2468_pairwise_spltfndr<FeatT, LabelT>::ncc_levels_2468_pairwise_spltfndr(const training_set<FeatT> & training_set,
                                                                                        const tset_indices_t& valid_indices,
                                                                                        const multivariate_normal<LabelT> & parent_dist,
                                                                                        training_set_idx_t num_in_parent,
                                                                                        uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
    : split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
        if (this->_feature_dimensionality != 600) {
            throw std::logic_error("Wrong dimensionality source data supplied to split finder, expected 600D.");
        }
        if ((num_splits_to_try % 4) != 0) {
            throw std::logic_error("num_splits_to_try must be divisible by 4");
        }
    }

    template<typename FeatT, typename LabelT>
    void ncc_levels_2468_pairwise_spltfndr<FeatT, LabelT>::calculate_vec_dot_prod_features(const garf_types<feature_idx_t>::vector & candidates_i,
                                                                                           const garf_types<feature_idx_t>::vector & candidates_j,
                                                                                           typename garf_types<FeatT>::matrix & feature_values) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_candidate_pairs = candidates_i.size();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        for (split_idx_t p = 0; p < num_candidate_pairs; p++) {
            // x and y offsets stored consecutively in memory
            feature_idx_t vec_i_x = 5 * candidates_i(p);
            feature_idx_t vec_i_y = vec_i_x + 1;
            feature_idx_t vec_j_x = 5 * candidates_j(p);
            feature_idx_t vec_j_y = vec_j_x + 1;
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                training_set_idx_t data_idx = this->_valid_indices(i);
                // do 2D dot product
                feature_values(i, p) = features(data_idx, vec_i_x) * features(data_idx, vec_j_x) + 
                                       features(data_idx, vec_i_y) * features(data_idx, vec_j_y);
            }
        }                                                                                                     
    }


    template<typename FeatT, typename LabelT>
    inf_gain_t ncc_levels_2468_pairwise_spltfndr<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                                    tset_indices_t** samples_going_right,
                                                                                    multivariate_normal<LabelT> & left_dist,
                                                                                    multivariate_normal<LabelT> & right_dist,
                                                                                    boost::random::mt19937 & gen) {
    
        // We are going to pick an equal amount from each level
        const feature_idx_t NUM_LEVELS = 4;

        // These are the start and end indices to put into the RNG for each level.
        // NOTE these are not indices into the actual feature vector, you must multiply
        // by 5 to get that because each ncc patch takes up 5 features in vector
        const feature_idx_t FEAT_RANGES[NUM_LEVELS][2] = {
            {0, 4}, // 4 arrows on level 2
            {4, 20}, // 16 on level 4
            {20, 56}, // 36 on level 6
            {56, 120} // 64 on level 8
        };

        // Sample uniformly from each level for the momemnt
        feature_idx_t num_level_2 = this->num_splits_to_try() / 4;
        feature_idx_t num_level_4 = this->num_splits_to_try() / 4;
        feature_idx_t num_level_6 = this->num_splits_to_try() / 4;
        feature_idx_t num_level_8 = this->num_splits_to_try() / 4;

        feature_idx_t level_4_offset = num_level_2;
        feature_idx_t level_6_offset = level_4_offset + num_level_4;
        feature_idx_t level_8_offset = level_6_offset + num_level_6;
        feature_idx_t level_8_limit = this->num_splits_to_try();

        // For each level, store the indices of the 2 vectors we are comparing
        typename garf_types<feature_idx_t>::vector candidates_i(this->num_splits_to_try());
        typename garf_types<feature_idx_t>::vector candidates_j(this->num_splits_to_try());

        // RNGs for each level
        uniform_int_distribution<> feature_chooser_level_2(FEAT_RANGES[0][0], FEAT_RANGES[0][1]);
        uniform_int_distribution<> feature_chooser_level_4(FEAT_RANGES[1][0], FEAT_RANGES[1][1]);
        uniform_int_distribution<> feature_chooser_level_6(FEAT_RANGES[2][0], FEAT_RANGES[2][1]);
        uniform_int_distribution<> feature_chooser_level_8(FEAT_RANGES[3][0], FEAT_RANGES[3][1]);

        // Generate the indices for level 2. Can't be arsed with 
        for (uint32_t i = 0; i < level_4_offset; i++) {
            candidates_i(i) = feature_chooser_level_2(gen);

            // Loop until we have generated a second candidate index which is different,
            // otherwise we compare a feature to itself which is boring
            do {
                candidates_j(i) = feature_chooser_level_2(gen);
            } while(candidates_i(i) == candidates_j(i)); 
        }

        // generate indices for level 4
        for (uint32_t i = level_4_offset; i < level_6_offset; i++) {
            candidates_i(i) = feature_chooser_level_4(gen);
            do {
                candidates_j(i) = feature_chooser_level_4(gen);
            } while (candidates_i(i) == candidates_j(i));
        }

        // generate indices for level 6
        for (uint32_t i = level_6_offset; i < level_8_offset; i++) {
            candidates_i(i) = feature_chooser_level_6(gen);
            do {
                candidates_j(i) = feature_chooser_level_6(gen);
            } while (candidates_i(i) == candidates_j(i));
        }

        // generate indices for level 8
        for (uint32_t i = level_8_offset; i < level_8_limit; i++) {
            candidates_i(i) = feature_chooser_level_8(gen);
            do {
                candidates_j(i) = feature_chooser_level_8(gen);
            } while (candidates_i(i) == candidates_j(i));
        }

#ifdef VERBOSE
        std::cout << "candidates chosen:\ni: " << candidates_i << "\nj: " << candidates_j << "\n";
#endif

        // Now we know which features we are comparing. Generate features for all of them
        typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples, num_level_2 + num_level_4 + num_level_6 + num_level_8);
        calculate_vec_dot_prod_features(candidates_i, candidates_j, feature_values); // zero offset as we are at the start of the array

#ifdef VERBOSE
        std::cout << "dot product features: " << feature_values << "\n";
#endif

        // Use the superclass functionality to say which split is best.
        inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                                                   samples_going_left, samples_going_right,
                                                                   left_dist, right_dist, gen);
        // store the x components, and the y is standardly x+1. Multiply in
        // the factor of 5 here, no point recomputing it on every evaluate()
        _feat_i_x = 5 * candidates_i(this->best_split_idx());
        _feat_j_x = 5 * candidates_j(this->best_split_idx());

        return best_information_gain;
    }

    template<typename FeatT, typename LabelT>
    inline split_direction_t ncc_levels_2468_pairwise_spltfndr<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & row) const {
        // std::cout << "inside evaluate()" << std::endl;

        // Remember, adding on one to the x coordinate feature index gets us the y coordinate feature index
        FeatT dot_product = row(_feat_i_x) * row(_feat_j_x) + 
                            row(_feat_i_x+1) * row(_feat_j_x+1);
        if (dot_product <= this->best_split_thresh()) {
            return LEFT;
        }
        else {
            return RIGHT;
        }
    }
}



#endif