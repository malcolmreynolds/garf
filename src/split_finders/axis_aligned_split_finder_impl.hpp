#ifndef GARF_AXIS_ALIGNED_SPLIT_FINDER_IMPL_HPP
#define GARF_AXIS_ALIGNED_SPLIT_FINDER_IMPL_HPP

namespace garf {

    template<typename FeatT, typename LabelT>
    axis_aligned_split_finder<FeatT, LabelT>::axis_aligned_split_finder(const training_set<FeatT> & training_set,
                                                                        const tset_indices_t & valid_indices,                                                                      
                                                                        const multivariate_normal<LabelT> & parent_dist,
                                                                        training_set_idx_t num_in_parent,
                                                                        uint32_t num_splits_to_try, uint32_t num_threshes_per_split) 
     : split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
           // std::cout << "axis_aligned_split_finder._valid_indices" << this->_valid_indices << std::endl;

        // If we have been asked to try more dimensions than there are, just try everything. Not that for axis aligned,
        // the number of splits we are trying is equivalent to the number of dimensions. For the general case, a split can
        // include a combination of 2 or more dimensions, so this functionality should not be put in the parent class.
        if (num_splits_to_try > this->_feature_dimensionality) {
            this->set_num_splits_to_try(this->_feature_dimensionality);
        }
    }

    // Once we have decided which features to test against, compute each feature for each bit of data that falls in the node. Also
    // useful for debugging, we can see the distribution of the features.
    template<typename FeatT, typename LabelT>
    void axis_aligned_split_finder<FeatT, LabelT>::calculate_all_features(const typename garf_types<feature_idx_t>::vector & dims_to_try, 
                                                                        typename garf_types<FeatT>::matrix & feature_values) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();                                                                        
        feature_idx_t num_features = dims_to_try.size();                                                                
        training_set_idx_t num_data_points = this->_num_training_samples;


        for (feature_idx_t f = 0; f < num_features; f++) {
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                feature_values(i,f) = features(this->_valid_indices(i), dims_to_try(f));
            }
        }                                                               
    }

    template<typename FeatT, typename LabelT>
    inf_gain_t axis_aligned_split_finder<FeatT, LabelT>::find_optimal_split(garf_types<training_set_idx_t>::vector** samples_going_left,
                                                                      garf_types<training_set_idx_t>::vector** samples_going_right,
                                                                      multivariate_normal<LabelT>& left_dist, 
                                                                      multivariate_normal<LabelT>& right_dist,
                                                                      boost::random::mt19937& gen) {
        // Pick which features to try and split    
        uniform_int_distribution<> dimension_chooser(0, this->_feature_dimensionality-1);
        typename garf_types<feature_idx_t>::vector dims_to_try(this->num_splits_to_try());
        for (uint32_t i=0; i < this->num_splits_to_try(); i++)  {
            dims_to_try(i) = dimension_chooser(gen);
        }
#ifdef VERBOSE
        std::cout << "feature dimensionality of " << this->_feature_dimensionality << ", examining feature indices " << dims_to_try << std::endl;
#endif

        //Evaluate each of the features for each data point
        typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples, this->num_splits_to_try());
        calculate_all_features(dims_to_try, feature_values);
#ifdef VERBOSE
        std::cout << "Evaluated all features:" << std::endl;
        print_matrix<FeatT>(std::cout, feature_values);
#endif

        inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                samples_going_left, samples_going_right,
                                left_dist, right_dist, gen);

        // calculate_information_gains finds us the best split index, but since we generated a random subset of features
        // in 'dims_to_try' we need to lookup into that.
        _split_feature = dims_to_try(this->best_split_idx());

#ifdef VERBOSE                    
        std::cout << "_split_feature = " << _split_feature << std::endl;
        std::cout << "calculated information gains, result is:" << std::endl
            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif    
        return best_information_gain;
    }

    // Perform the inner product to see which side of the hyperplane we are on
    template<typename FeatT, typename LabelT>
    inline split_direction_t axis_aligned_split_finder<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const {
        if (sample(_split_feature) <= this->best_split_thresh()) {
            return LEFT;
        }
        return RIGHT;
    }
}



#endif