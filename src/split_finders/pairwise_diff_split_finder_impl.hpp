#ifndef GARF_PAIRWISE_DIFF_SPLIT_FINDER_IMPL_HPP
#define GARF_PAIRWISE_DIFF_SPLIT_FINDER_IMPL_HPP

namespace garf {
	
	template<typename FeatT, typename LabelT>
	pairwise_diff_split_finder<FeatT, LabelT>::pairwise_diff_split_finder(const training_set<FeatT> & training_set,
										                                  const tset_indices_t& valid_indices,
                                                                          const multivariate_normal<LabelT> & parent_dist,
                                                                          training_set_idx_t num_in_parent,
										                                  uint32_t num_splits_to_try, uint32_t num_threshes_per_split) 
	: split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
		if (this->_feature_dimensionality == 1) {
			throw std::logic_error("Pairwise comparison split finder is useless for 1D data!");
		}
	}


    template<typename FeatT, typename LabelT>
    void pairwise_diff_split_finder<FeatT, LabelT>::calculate_all_features(const typename garf_types<feature_idx_t>::vector & feat_i_candidates,
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
    inf_gain_t pairwise_diff_split_finder<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                       tset_indices_t** samples_going_right,
                                                                       multivariate_normal<LabelT>& left_dist, 
                                                                       multivariate_normal<LabelT>& right_dist,
                                                                       boost::random::mt19937& gen) {

        // Need to generate a pair of indices for each candidate split
        typename garf_types<feature_idx_t>::vector feat_i_candidates(this->num_splits_to_try());
        typename garf_types<feature_idx_t>::vector feat_j_candidates(this->num_splits_to_try());
        uniform_int_distribution<> dimension_chooser(0, this->_feature_dimensionality-1);

        // Generate random pairs of feature indices, making sure not to have any pairs comparing 
        // one feature to itself, as that obviously doesn't do anything useful.
        for (uint32_t i=0; i < this->num_splits_to_try(); i++) {
        	while (true) {
        		feat_i_candidates(i) = dimension_chooser(gen);
        		feat_j_candidates(i) = dimension_chooser(gen);
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
    inline split_direction_t pairwise_diff_split_finder<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & row) const {
    	if ((row(_feat_i) - row(_feat_j)) <= this->best_split_thresh()) {
    		return LEFT;
    	}
    	return RIGHT;
    }
}



#endif