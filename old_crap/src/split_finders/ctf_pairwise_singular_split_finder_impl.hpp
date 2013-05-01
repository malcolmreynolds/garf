#ifndef GARF_CTF_PAIRWISE_SINGULAR_SPLIT_FINDER_IMPL_HPP
#define GARF_CTF_PAIRWISE_SINGULAR_SPLIT_FINDER_IMPL_HPP

namespace garf {
	
	template<typename FeatT, typename LabelT>
	ctf_pairwise_singular_split_finder<FeatT, LabelT>::ctf_pairwise_singular_split_finder(const training_set<FeatT> & training_set,
										                                  const tset_indices_t& valid_indices,
                                                                          const multivariate_normal<LabelT> & parent_dist,
                                                                          training_set_idx_t num_in_parent,
										                                  uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
	: split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
		if (this->_feature_dimensionality != 1360) {
			throw std::logic_error("Wrong dimensionality source data supplied to split finder.");
		}
	}


	// Fill the entires of feature_values. We put stuff in with feat_val_offset as an index offset, to allow one matrix to store both the
	// pairwise and singular choices
	template<typename FeatT, typename LabelT>
    void ctf_pairwise_singular_split_finder<FeatT, LabelT>::calculate_pairwise_diff_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
									                                                     const garf_types<feature_idx_t>::vector & feat_j_candidates,
									                                                     typename garf_types<FeatT>::matrix & feature_values,
									                                                     uint32_t feat_val_offset) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_candidate_pairs = feat_i_candidates.size();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        for (split_idx_t p = 0; p < num_candidate_pairs; p++) {
        	feature_idx_t feat_i_idx = feat_i_candidates(p);
        	feature_idx_t feat_j_idx = feat_j_candidates(p);
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
            	training_set_idx_t data_idx = this->_valid_indices(i);
                feature_values(i, p + feat_val_offset) = features(data_idx, feat_i_idx) - features(data_idx, feat_j_idx);
            }
        }                                                                           
    }


    // Fill the entires of feature_values. We put stuff in with feat_val_offset as an index offset, to allow one matrix to store both the
    // pairwise and singular choices
    template<typename FeatT, typename LabelT>
    void ctf_pairwise_singular_split_finder<FeatT, LabelT>::calculate_pairwise_plus_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
                                                                                          const garf_types<feature_idx_t>::vector & feat_j_candidates,
                                                                                          typename garf_types<FeatT>::matrix & feature_values,
                                                                                          uint32_t feat_val_offset) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_candidate_pairs = feat_i_candidates.size();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        for (split_idx_t p = 0; p < num_candidate_pairs; p++) {
            feature_idx_t feat_i_idx = feat_i_candidates(p);
            feature_idx_t feat_j_idx = feat_j_candidates(p);
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                training_set_idx_t data_idx = this->_valid_indices(i);
                feature_values(i, p + feat_val_offset) = features(data_idx, feat_i_idx) + features(data_idx, feat_j_idx);
            }
        }                                                                           
    }

    // Fill the entires of feature_values. We put stuff in with feat_val_offset as an index offset, to allow one matrix to store both the
    // pairwise and singular choices
    template<typename FeatT, typename LabelT>
    void ctf_pairwise_singular_split_finder<FeatT, LabelT>::calculate_pairwise_abs_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
                                                                                          const garf_types<feature_idx_t>::vector & feat_j_candidates,
                                                                                          typename garf_types<FeatT>::matrix & feature_values,
                                                                                          uint32_t feat_val_offset) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_candidate_pairs = feat_i_candidates.size();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        for (split_idx_t p = 0; p < num_candidate_pairs; p++) {
            feature_idx_t feat_i_idx = feat_i_candidates(p);
            feature_idx_t feat_j_idx = feat_j_candidates(p);
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                training_set_idx_t data_idx = this->_valid_indices(i);
                feature_values(i, p + feat_val_offset) = fabs(features(data_idx, feat_i_idx) - features(data_idx, feat_j_idx));
            }
        }                                                                           
    }

    template<typename FeatT, typename LabelT>
    void ctf_pairwise_singular_split_finder<FeatT, LabelT>::calculate_singular_features(const typename garf_types<feature_idx_t>::vector & singular_feat_candidates,
                                                                                        typename garf_types<FeatT>::matrix & feature_values,
                                                                                        uint32_t feat_val_offset) const {
		const typename garf_types<FeatT>::matrix & features = this->_training_set.features();
		split_idx_t num_singular_candidates = singular_feat_candidates.size();
		training_set_idx_t num_data_points = this->_num_training_samples;
		
		for (split_idx_t p = feat_val_offset; p < num_singular_candidates; p++) {
			for (training_set_idx_t i = 0; i < num_data_points; i++) {
				feature_values(i, p + feat_val_offset) = features(this->_valid_indices(i), singular_feat_candidates(p));
			}
		}
    }

    template<typename FeatT, typename LabelT>
    inf_gain_t ctf_pairwise_singular_split_finder<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                                     tset_indices_t** samples_going_right,
                                                                                     multivariate_normal<LabelT> & left_dist,
                                                                                     multivariate_normal<LabelT> & right_dist,
                                                                                     boost::random::mt19937 & gen) {
    
    	const feature_idx_t NUM_RESOLUTIONS = 4;
    	
    	const feature_idx_t FEATURE_RANGES[NUM_RESOLUTIONS][3] = {
            {0, 8, 16}, // 2x2 resolution = 8 means, 8 variances
            {16, 48, 80}, // 4x4 resolution = 32 means , 32 variances
            {80, 208, 336}, // 8x8 resolution = 128 means, 128 variances
            {336, 848, 1360} // 16 x 16 resolution = 512 means, 512 variances
    	};

    	// Do same number of each
    	feature_idx_t num_singular = this->num_splits_to_try() / 4;
    	feature_idx_t num_pair_diff = this->num_splits_to_try() / 4;
        feature_idx_t num_pair_plus = this->num_splits_to_try() / 4;
        feature_idx_t num_pair_abs = this->num_splits_to_try() / 4;

#ifdef VERBOSE
        std::cout << num_singular << " singular splits, " << num_pair_diff << " pws_diff splits, "
                  << num_pair_plus << " pws_plus splits, " << num_pair_abs << " pwd_abs splits" << std::endl;
#endif
        // candidates for feature i > threshold
    	typename garf_types<feature_idx_t>::vector singular_candidates(num_singular);

        // candidates for feature i - feature_j > threshold
    	typename garf_types<feature_idx_t>::vector feat_i_diff(num_pair_diff);
        typename garf_types<feature_idx_t>::vector feat_j_diff(num_pair_diff); 

        //candidates for feature i + feature_j > threshold
        typename garf_types<feature_idx_t>::vector feat_i_plus(num_pair_plus);
        typename garf_types<feature_idx_t>::vector feat_j_plus(num_pair_plus);

        //candidates for abs(feature i - feature j) > threshold
        typename garf_types<feature_idx_t>::vector feat_i_abs(num_pair_abs);
        typename garf_types<feature_idx_t>::vector feat_j_abs(num_pair_abs);

        // First gen to pick the resolution we work at
        uniform_int_distribution<> resolution_chooser(0, NUM_RESOLUTIONS-1);

        // Just gives a random boolean, to pick whether to examine mean or variance
        uniform_int_distribution<> mean_var_chooser(0, 1);

        // Need several other RNGs to pick within each range - the second index selects mean
        // or variance. FIXME: this is a horrible pile of crap, but see if it works before
        // optimising anything
        uniform_int_distribution<> sub_rngs[NUM_RESOLUTIONS][2] = {
        	{uniform_int_distribution<>(FEATURE_RANGES[0][0],
	        	                        FEATURE_RANGES[0][1]-1),
	         uniform_int_distribution<>(FEATURE_RANGES[0][1],
	                                    FEATURE_RANGES[0][2]-1)},
        	{uniform_int_distribution<>(FEATURE_RANGES[1][0],
	        	                        FEATURE_RANGES[1][1]-1),
	         uniform_int_distribution<>(FEATURE_RANGES[1][1],
	                                    FEATURE_RANGES[1][2]-1)},	                   
        	{uniform_int_distribution<>(FEATURE_RANGES[2][0],
	        	                        FEATURE_RANGES[2][1]-1),
	         uniform_int_distribution<>(FEATURE_RANGES[2][1],
	                                    FEATURE_RANGES[2][2]-1)},
        	{uniform_int_distribution<>(FEATURE_RANGES[3][0],
	        	                        FEATURE_RANGES[3][1]-1),
	         uniform_int_distribution<>(FEATURE_RANGES[3][1],
	                                    FEATURE_RANGES[3][2]-1)}
	    };

	    // Genereate some singular candidates.. 
	    for (uint32_t i = 0; i < num_singular; i++) {
	    	feature_idx_t res = resolution_chooser(gen);
	    	feature_idx_t mean_or_var = mean_var_chooser(gen);
            singular_candidates(i) = sub_rngs[res][mean_or_var](gen);
 	    }

	    //.. and some pairwise candidates
        for (uint32_t i=0; i < num_pair_diff; i++) {
        	// Pick a resolution
        	feature_idx_t res = resolution_chooser(gen);
        	// pick mean or variance
        	feature_idx_t mean_or_var = mean_var_chooser(gen);
        	while (true) {
        		feat_i_diff(i) = sub_rngs[res][mean_or_var](gen);
        		feat_j_diff(i) = sub_rngs[res][mean_or_var](gen);
        		if (feat_i_diff(i) != feat_j_diff(i)) {
        			break;
        		}
            }
        }

        // candidates for plus features
        for (uint32_t i=0; i < num_pair_plus; i++) {
            // Pick a resolution
            feature_idx_t res = resolution_chooser(gen);
            // pick mean or variance
            feature_idx_t mean_or_var = mean_var_chooser(gen);
            while (true) {
                feat_i_plus(i) = sub_rngs[res][mean_or_var](gen);
                feat_j_plus(i) = sub_rngs[res][mean_or_var](gen);
                if (feat_i_plus(i) != feat_j_plus(i)) {
                    break;
                }
            }
        }

        // candidates for abs features
        for (uint32_t i=0; i < num_pair_abs; i++) {
            // Pick a resolution
            feature_idx_t res = resolution_chooser(gen);
            // pick mean or variance
            feature_idx_t mean_or_var = mean_var_chooser(gen);
            while (true) {
                feat_i_abs(i) = sub_rngs[res][mean_or_var](gen);
                feat_j_abs(i) = sub_rngs[res][mean_or_var](gen);
                if (feat_i_abs(i) != feat_j_abs(i)) {
                    break;
                }
            }
        }

#ifdef VERBOSE
        // std::cout << "feature dimensionality of " << this->_feature_dimensionality
        // 	<< ", examining feature pairs i: " << feat_i_candidates << std::endl
        // 	<< " j: " << feat_j_candidates << std::endl
        // 	<< "and singular features: " << singular_candidates << std::endl;
#endif

		typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples, num_singular + num_pair_diff + num_pair_plus + num_pair_abs);
		calculate_singular_features(singular_candidates, feature_values, 0); // zero offset as we are at the start of the array
		calculate_pairwise_diff_features(feat_i_diff, feat_j_diff, feature_values, num_singular);
        calculate_pairwise_plus_features(feat_i_plus, feat_j_plus, feature_values, num_singular + num_pair_diff);
        calculate_pairwise_abs_features(feat_i_abs, feat_j_abs, feature_values, num_singular + num_pair_diff + num_pair_plus);

#ifdef VERBOSE
        std::cout << "Evaluated all features:" << std::endl;
        print_matrix<FeatT>(std::cout, feature_values);
#endif

		inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                    		                       samples_going_left, samples_going_right,
                                    		                       left_dist, right_dist, gen);
		if (this->best_split_idx() < num_singular) {
			_feat_i = singular_candidates(this->best_split_idx());
			_type = SINGULAR;
#ifdef VERBOSE                    
	        std::cout << "Found best split is singular: " << _feat_i << std::endl;
	        std::cout << "calculated information gains, result is:" << std::endl
	            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
	            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif   
		}
        //pairwise diff chosen
        else if (this->best_split_idx() < (num_singular + num_pair_diff)) {
			_feat_i = feat_i_diff(this->best_split_idx() - num_singular);
			_feat_j = feat_j_diff(this->best_split_idx() - num_singular);
            _type = PAIR_DIFF;
#ifdef VERBOSE                    
	        std::cout << "Found best pair: (" << _feat_i << "," << _feat_j << ")" << std::endl;
	        std::cout << "calculated information gains, result is:" << std::endl
	            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
	            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif   
		}
        else if (this->best_split_idx() < (num_singular + num_pair_diff + num_pair_plus)) { //pairwise plus chosen
            uint32_t offset = num_singular + num_pair_diff;
            _feat_i = feat_i_plus(this->best_split_idx() - offset);
            _feat_j = feat_j_plus(this->best_split_idx() - offset);
            _type = PAIR_PLUS;
        }
        else { 
            uint32_t offset = num_singular + num_pair_diff + num_pair_plus;
            _feat_i = feat_i_abs(this->best_split_idx() - offset);
            _feat_j = feat_j_abs(this->best_split_idx() - offset);
            _type = PAIR_ABS;
        }
    
		return best_information_gain;
    }


    template<typename FeatT, typename LabelT>
    inline split_direction_t ctf_pairwise_singular_split_finder<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & row) const {
    	// std::cout << "inside evaluate()" << std::endl;
        switch(_type) {
            case PAIR_DIFF:
                if ((row(_feat_i) - row(_feat_j)) <= this->best_split_thresh()) {
                    return LEFT;
                }
                else {
                    return RIGHT;
                }
            case PAIR_PLUS:
                if ((row(_feat_i) + row(_feat_j)) <= this->best_split_thresh()) {
                    return LEFT;
                }
                else {
                    return RIGHT;
                }
            case PAIR_ABS:
                if (fabs(row(_feat_i) - row(_feat_j)) <= this->best_split_thresh()) {
                    return LEFT;
                }
                else {
                    return RIGHT;
                }
            case SINGULAR:
                if (row(_feat_i) <= this->best_split_thresh()) {
                    return LEFT;
                }
                else {
                    return RIGHT;
                }   
            default:
                throw std::logic_error("shouldn't be here - _type must be uninitialised.");
        }
    }
}




#endif