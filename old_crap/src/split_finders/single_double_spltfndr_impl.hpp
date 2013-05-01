#ifndef SINGLE_DOUBLE_SPLTFNDR_IMPL_HPP
#define SINGLE_DOUBLE_SPLTFNDR_IMPL_HPP

namespace garf {

    template<typename FeatT, typename LabelT>
    single_double_spltfndr<FeatT, LabelT>::single_double_spltfndr(const training_set<FeatT> & training_set,
                           const tset_indices_t& valid_indices,
                           const multivariate_normal<LabelT> & parent_dist,
                           training_set_idx_t num_in_parent,
                           uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
        : split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
    }

    template<typename FeatT, typename LabelT>
    void single_double_spltfndr<FeatT, LabelT>::calculate_singular_features(const typename garf_types<feature_idx_t>::vector & dims_to_try, 
                                                                            typename garf_types<FeatT>::matrix & feature_values) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();                                                                        
        feature_idx_t num_features = dims_to_try.size();                                                                
        training_set_idx_t num_data_points = this->_num_training_samples;

#ifdef VERBOSE
        std::cout << "inside calculate_singular_features, num_features = " 
            << num_features << " num_data_points = " 
            << num_data_points << std::endl;
#endif

        for (feature_idx_t f = 0; f < num_features; f++) {
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                feature_values(i, f) = features(this->_valid_indices(i), dims_to_try(f));
                // std::cout << "feature_values(" << i << "," << f << ") = " 
                //     << feature_values(i, f) << std::endl;
            }
        } 
    }

    template<typename FeatT, typename LabelT>
    void single_double_spltfndr<FeatT, LabelT>::calculate_pairwise_features(const typename garf_types<feature_idx_t>::matrix & pairwise_dims,
                                                                            const typename garf_types<FeatT>::matrix & pairwise_coeffs,
                                                                            typename garf_types<FeatT>::matrix & feature_values,
                                                                            split_idx_t offset) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();                                                                        
        feature_idx_t num_features = pairwise_dims.size2();                                                                
        training_set_idx_t num_data_points = this->_num_training_samples;

#ifdef VERBOSE
        std::cout << "inside calculate_pairise_features, num_features = "
            << num_features << " num_data_points = " 
            << num_data_points << " offset = "
            << offset << std::endl;
#endif

        FeatT hyperplane_value;
        for (feature_idx_t f = 0; f < num_features; f++) {
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                // Extract which bit of data we are interested in
                training_set_idx_t this_data_idx = this->_valid_indices(i);

                // Hyperplnae involves extracting two elements from feature matrix, 
                // multiplying them by the relevant coefficients
                hyperplane_value = (features(this_data_idx, pairwise_dims(0, f))
                                     * pairwise_coeffs(0, f));
                hyperplane_value += (features(this_data_idx, pairwise_dims(1, f))
                                     * pairwise_coeffs(1, f));
                feature_values(i, f + offset) = hyperplane_value;
            }
        } 
    }


    template<typename FeatT, typename LabelT>
    inf_gain_t single_double_spltfndr<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                         tset_indices_t** samples_going_right,
                                                                         multivariate_normal<LabelT>& left_dist, 
                                                                         multivariate_normal<LabelT>& right_dist,
                                                                         boost::random::mt19937& gen) {
        feature_idx_t num_singular_to_try = 3*this->num_splits_to_try() / 4;
        feature_idx_t num_pairwise_to_try = this->num_splits_to_try() - num_singular_to_try;

        // Pick which features to try and split    
        typename garf_types<feature_idx_t>::vector single_dims_to_try(num_singular_to_try);
        typename garf_types<feature_idx_t>::matrix pairwise_dims_to_try(2, num_pairwise_to_try);
        // coeffs to allow the 2D hyper(sub)plane to be steered
        typename garf_types<FeatT>::matrix pairwise_coeffs(2, num_pairwise_to_try); 

        // Create the necessary Random Number Generators
        uniform_int_distribution<> dimension_chooser(0, this->_feature_dimensionality-1);
        normal_distribution<> coeff_chooser(0.0, 1.0);

        // Generate 
        for (split_idx_t i = 0; i < num_singular_to_try; i++)  {
            single_dims_to_try(i) = dimension_chooser(gen);
        }

        // This only makes one coefficient per pick of indices, kind of sucks but hey,
        // much easier to implement for the moment
        for (split_idx_t j = 0; j < 2; j++) {    
            for (split_idx_t i = 0; i < num_pairwise_to_try; i++) {
                pairwise_dims_to_try(j, i) = dimension_chooser(gen);
                pairwise_coeffs(j, i) = coeff_chooser(gen);
            }
        }

#ifdef VERBOSE
        std::cout << "feature dimensionality of " << this->_feature_dimensionality 
            << ", examining singular feature indices " << single_dims_to_try
            << " and pairwise feature indices " << pairwise_dims_to_try << std::endl;
#endif

        // Evaluate both the pairwise and singular splits for every datapoint
        typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples,
                                                          num_singular_to_try + num_pairwise_to_try);
        calculate_singular_features(single_dims_to_try, feature_values);
        calculate_pairwise_features(pairwise_dims_to_try, pairwise_coeffs, feature_values, num_singular_to_try);

#ifdef VERBOSE
        std::cout << "Evaluated all features:" << std::endl;
        print_matrix<FeatT>(std::cout, feature_values);
#endif

        inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                samples_going_left, samples_going_right,
                                left_dist, right_dist, gen);

        // calculate_information_gains finds us the best split index, but since
        // we generated a random subset of features in 'dims_to_try'
        // we need to lookup into that.
        split_idx_t chosen_split = this->best_split_idx();
        if (chosen_split < num_singular_to_try) {
            _type = SINGLE;
            _feat_i = single_dims_to_try(chosen_split);
        }
        else {
            _type = DOUBLE;
            // Need to convert from an offset into all the evaluated splits to
            // and offset into just the pairwise ones.
            chosen_split -= num_singular_to_try; 
            _feat_i = pairwise_dims_to_try(0, chosen_split);
            _feat_j = pairwise_dims_to_try(1, chosen_split);
            _coeff_i = pairwise_coeffs(0, chosen_split);
            _coeff_j = pairwise_coeffs(1, chosen_split);
        }

#ifdef VERBOSE        
        if (_type == SINGLE){
            std::cout << "_type=SINGLE, _feat_i = " << _feat_i << std::endl;
        }
        else if (_type == DOUBLE){
            std::cout << "_type=DOUBLE, features = (" << _feat_i << "," << _feat_j << ")"
                << " coeffs = (" << _coeff_i << "," << _coeff_j << ") " << std::endl;
        }
        std::cout << "calculated information gains, result is:" << std::endl
            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif    
        return best_information_gain;
    }


    // Perform the inner product to see which side of the hyperplane we are on
    template<typename FeatT, typename LabelT>
    inline split_direction_t single_double_spltfndr<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const {
        // This might be either an axis aligned split or a hyperplane splittin across a
        // 2d subplane
        FeatT result_feature; // build result in here
        if (_type == SINGLE) {
            result_feature = sample(_feat_i);
        }
        else if (_type == DOUBLE) {
            result_feature = (sample(_feat_i) * _coeff_i) + 
                             (sample(_feat_j) * _coeff_j);
        }
        else {
            throw std::logic_error("_type is not set to either SINGLE or DOUBLE");
        }

        if (result_feature <= this->best_split_thresh()) {
            return LEFT;
        }
        return RIGHT;
    }
}



#endif