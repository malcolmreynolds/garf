#ifndef GARF_ctf_pairwise_only_split_finder_IMPL_HPP
#define GARF_ctf_pairwise_only_split_finder_IMPL_HPP

namespace garf {


	template<typename FeatT, typename LabelT>
	ctf_pairwise_only_split_finder<FeatT, LabelT>::ctf_pairwise_only_split_finder(const training_set<FeatT> & training_set,
										                                  const tset_indices_t& valid_indices,
                                                                          const multivariate_normal<LabelT> & parent_dist,
                                                                          training_set_idx_t num_in_parent,
										                                  uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
	: split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
		if (this->_feature_dimensionality != 1360) {
			throw std::logic_error("Wrong dimensionality source data supplied to split finder.");
		}
	}


    template<typename FeatT, typename LabelT>
    void ctf_pairwise_only_split_finder<FeatT, LabelT>::calculate_all_features(const typename garf_types<feature_idx_t>::vector & feat_i_candidates,
									                                       const typename garf_types<feature_idx_t>::vector & feat_j_candidates,
									                                       typename garf_types<FeatT>::matrix & feature_values) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_candidate_pairs = feat_i_candidates.size();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        for (split_idx_t p = 0; p < num_candidate_pairs; p++) {
        	feature_idx_t feat_i_idx = feat_i_candidates(p);
        	feature_idx_t feat_j_idx = feat_j_candidates(p);
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
            	training_set_idx_t data_idx = this->_valid_indices(i);
                feature_values(i, p) = features(data_idx, feat_i_idx) - features(data_idx, feat_j_idx);
            }
        }                                                                           
    }


    template<typename FeatT, typename LabelT>
    inf_gain_t ctf_pairwise_only_split_finder<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                       tset_indices_t** samples_going_right,
                                                                       multivariate_normal<LabelT>& left_dist, 
                                                                       multivariate_normal<LabelT>& right_dist,
                                                                       boost::random::mt19937& gen) {

        // I would initialise                                                               
        const feature_idx_t NUM_RESOLUTIONS = 4;                                                                       		

        const feature_idx_t FEATURE_RANGES[NUM_RESOLUTIONS][3] = {
            {0, 8, 16}, // 2x2 resolution = 8 means, 8 variances
            {16, 48, 80}, // 4x4 resolution = 32 means , 32 variances
            {80, 208, 336}, // 8x8 resolution = 128 means, 128 variances
            {336, 848, 1360} // 16 x 16 resolution = 512 means, 512 variances
        };

        // Need to generate a pair of indices for each candidate split
        typename garf_types<feature_idx_t>::vector feat_i_candidates(this->num_splits_to_try());
        typename garf_types<feature_idx_t>::vector feat_j_candidates(this->num_splits_to_try());

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

        // Generate random pairs of feature indices, making sure not to have any pairs comparing 
        // one feature to itself, as that obviously doesn't do anything useful.

        // THIS LOOP is basically the only difference from the previous iteration
        for (uint32_t i=0; i < this->num_splits_to_try(); i++) {
        	// Pick a resolution
        	feature_idx_t res = resolution_chooser(gen);
        	// pick mean or variance
        	feature_idx_t mean_or_var = mean_var_chooser(gen);
        	while (true) {
        		feat_i_candidates(i) = sub_rngs[res][mean_or_var](gen);
        		feat_j_candidates(i) = sub_rngs[res][mean_or_var](gen);
        		if (feat_i_candidates(i) != feat_j_candidates(i)) {
        			break;
        		}
            }
        }
#ifdef VERBOSE
        std::cout << "feature dimensionality of " << this->_feature_dimensionality
        	<< ", examining feature pairs i: " << feat_i_candidates << std::endl
        	<< " j: " << feat_j_candidates << std::endl;
#endif

        //Evaluate each of the features for each data point
        typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples, this->num_splits_to_try());
        calculate_all_features(feat_i_candidates, feat_j_candidates, feature_values);
#ifdef VERBOSE
        std::cout << "Evaluated all features:" << std::endl;
        print_matrix<FeatT>(std::cout, feature_values);
#endif

		inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                    		                       samples_going_left, samples_going_right,
                                    		                       left_dist, right_dist, gen);

        // At this point, calculate_information_gains has picked the index of the best pair
        _feat_i = feat_i_candidates(this->best_split_idx());
        _feat_j = feat_j_candidates(this->best_split_idx());

#ifdef VERBOSE                    
        std::cout << "Found best pair: (" << _feat_i << "," << _feat_j << ")" << std::endl;
        std::cout << "calculated information gains, result is:" << std::endl
            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif    
        return best_information_gain;
    }


    // Take the pairwise difference to see which side of the split we go on
    template<typename FeatT, typename LabelT>
    inline split_direction_t ctf_pairwise_only_split_finder<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & row) const {
    	if ((row(_feat_i) - row(_feat_j)) <= this->best_split_thresh()) {
    		return LEFT;
    	}
    	return RIGHT;
    }
}
#endif