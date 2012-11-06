#ifndef GARF_SPLIT_FINDER_IMPL_HPP
#define GARF_SPLIT_FINDER_IMPL_HPP

namespace garf {

    // the only thing really common to everything (so we put in parent class) is that we need to know the
    // dimensionality of data we are dealing with and how many splits to attempt
    template<typename FeatT, typename LabelT>
    split_finder<FeatT, LabelT>::split_finder(const training_set<FeatT> & training_set,
                                              const tset_indices_t & valid_indices,
                                              const multivariate_normal<LabelT> & parent_dist,
                                              training_set_idx_t num_in_parent,
                                              uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
          : _num_splits_to_try(num_splits_to_try),
            _num_threshes_per_split(num_threshes_per_split),
            _training_set(training_set),
            _valid_indices(valid_indices),
            _parent_dist(parent_dist),
            _num_in_parent(num_in_parent),
            _feature_dimensionality(training_set.feature_dimensionality()),
            _inf_gain_dimensionality(training_set.inf_gain_dimensionality()),
            _num_training_samples(valid_indices.size()) {
    }

    
    // After each feature has been evaluated, work out the mins and maxs so we know what values make sensible thresholds
    template<typename FeatT, typename LabelT>
    void split_finder<FeatT, LabelT>::calculate_mins_and_maxs(const typename garf_types<FeatT>::matrix& feature_values, 
                                                              typename garf_types<FeatT>::vector& feature_mins,
                                                              typename garf_types<FeatT>::vector& feature_maxs) const {
        // We are not calculating min and max for every feature, just the ones we have randomly selected
        training_set_idx_t num_data_points_here = feature_values.size1();
        feature_idx_t num_features = feature_values.size2();


        for (feature_idx_t f = 0; f < num_features; f++) {
            // std::cout << "doing feature " << f << std::endl;
            feature_mins(f) = feature_values(0, f);
            feature_maxs(f) = feature_values(0, f);

            // std::cout << "min/max initialised to " << _feature_mins << " / " << _feature_maxs << std::endl;

            for (training_set_idx_t i = 1; i < num_data_points_here; i++) {
                // std::cout << "sample_idx = " << sample_idx << std::endl;
                feature_mins(f) = std::min(feature_mins(f), feature_values(i,f));
                feature_maxs(f) = std::max(feature_maxs(f), feature_values(i,f));
            }
        }
    #ifdef VERBOSE
        std::cout << "mins = " << feature_mins << std::endl
            << "maxs = " << feature_maxs << std::endl;
    #endif
    }
    
    // Given some dimensions to try and split, fill a 'thresholds' matrix with a variety of splits to try,
    // sampled uniformley between the min and max of that feature range
    template<typename FeatT, typename LabelT>
    void split_finder<FeatT, LabelT>::generate_thresholds(const typename garf_types<FeatT>::vector& feature_mins,
                                                          const typename garf_types<FeatT>::vector& feature_maxs,
                                                          typename garf_types<FeatT>::matrix& thresholds,
                                                          boost::random::mt19937& gen) const {
        for (uint32_t feat_idx = 0; feat_idx < this->_num_splits_to_try; feat_idx++) {
    #ifdef VERBOSE        
            std::cout << "#" << feat_idx << " generating threshes between " << feature_mins(feat_idx) << " and " << feature_maxs(feat_idx) << std::endl;
    #endif

            // if there is no variation in the feature then the random sampler gets caught in an endless loop.. 
            // obviously there is nothing useful we can do when the feature doesn't vary, so just use constant
            // thresholds, which will result in nothing useful
            if (feature_mins(feat_idx) == feature_maxs(feat_idx)) {
                std::cout << "Warning: No variation in feature #" << feat_idx << std::endl;
                for (uint32_t j=0; j < this->_num_threshes_per_split; j++) {
                    thresholds(feat_idx, j) = feature_mins(feat_idx);
                }

                continue;
            }


    //        std::cout << "creating dimensions sampler between " << feature_mins(feat_idx) << " and " << feature_maxs(feat_idx) << std::endl;
            uniform_real_distribution<> dimension_sampler(feature_mins(feat_idx), feature_maxs(feat_idx));

            for (uint32_t j=0; j < this->_num_threshes_per_split; j++) {
                thresholds(feat_idx, j) = dimension_sampler(gen);
            }
        }
    }
    
    // For a given dimension, threshold, and combination of feature values and valid indices, returns the directions.
    // Note that here we do receive a feature matrix and indices as paramters, as they may not be the same as the ones we trained on 
    // (ie this function might be used as test time).
    // ALSO note that the split_feature_values is stuff that has been encoded by whatever our specialisation of split_finder is -
    // ie we have already done the hyperplane calculation to work out how far away on each side of the hyperplane the data is.
    // We just need to do the 1d threshold test on these values, which means we can put this is the base class
    template<typename FeatT, typename LabelT>
    void split_finder<FeatT, LabelT>::evaluate_single_split(const typename garf_types<FeatT>::matrix& split_feature_values, 
                                                            garf_types<split_direction_t>::vector& split_directions,
                                                            split_idx_t split_feature, FeatT split_value) {
        uint32_t num_samples_to_evaluate = split_feature_values.size1();
        if (split_directions.size() != num_samples_to_evaluate) {
            throw new std::invalid_argument("split_direction size doesn't match number of samples we are splitting on");
        }
        // std::cout << "evaluating axis aligned split on " << num_samples_to_evaluate << " samples" << std::endl;

        for (uint32_t i = 0; i < num_samples_to_evaluate; i++) {
            // Compare the chosen feature for datapoint i to the split value, write output direction
            if (split_feature_values(i, split_feature) <= split_value) {
                split_directions(i) = LEFT;
            }
            else {
                split_directions(i) = RIGHT;
            }
        }
        // std::cout << "returning from evaluate_single_split" << std::endl;
    }    
    
    
    // This calculates the information gain regardless of what kind of split we are doing, because the splits have already been performed and put in
    // the feature_values matrix
    template<typename FeatT, typename LabelT>
    inf_gain_t split_finder<FeatT, LabelT>::calculate_information_gains(const typename garf_types<FeatT>::matrix& thresholds,
                                                                  const typename garf_types<FeatT>::matrix& feature_values,
                                                                  // arguments below are the parameters of the best split, once we
                                                                  // have found it. They may be written to multiple times over the
                                                                  // course of the search.
                                                                  garf_types<training_set_idx_t>::vector** samples_going_left,
                                                                  garf_types<training_set_idx_t>::vector** samples_going_right,
                                                                  multivariate_normal<LabelT> & left_dist_out, 
                                                                  multivariate_normal<LabelT> & right_dist_out) {
        //keep track of the best scoring split we find here. Initialise at negative infinity                                                                             
        double best_information_gain = -std::numeric_limits<double>::infinity();
        double gain;
        bool found_a_good_gain = false;

        // these distributions will be used throughout
        multivariate_normal<LabelT> candidate_left_dist(this->_inf_gain_dimensionality);
        multivariate_normal<LabelT> candidate_right_dist(this->_inf_gain_dimensionality);
        garf_types<split_direction_t>::vector candidate_split_directions(this->_num_training_samples);

        // We allocate these to the maximum size they can possibly be, namely (num_samples_in_parent - 1)
        // as we don't bother doing anything further with a candidate split which doesn't separate the data at all.
        // This is slightly memory-ineffecient, but it does allow us to avoid reallocating for each split
        garf_types<training_set_idx_t>::vector candidate_samples_going_left(this->_num_training_samples-1);
        garf_types<training_set_idx_t>::vector candidate_samples_going_right(this->_num_training_samples-1);

        // These two coefficients will always sum to 1. Basically this means that at the top of the tree
        // we bias much more towards splits that evenly divide the data, and then further down we care more
        // about accuracy
        double coeff = pow(this->_balance_bias, this->_depth);
        double entropy_adjuster = (1 - coeff);
        double balance_adjuster = coeff;



        for (uint32_t i = 0; i < this->_num_splits_to_try; i++) {
            for (uint32_t j=0; j < this->_num_threshes_per_split; j++) {
                // Work out which data goes left and which goes right
#ifdef VERBOSE
                std::cout << "evaluating split at thresh " << thresholds(i,j) << " for feature index " << i << std::endl;
#endif
                evaluate_single_split(feature_values, candidate_split_directions, i, thresholds(i, j));

                // create index vectors for left and right. I considered putting these on the stack but they could be quite big
                // FIXME: allocate these outside the loop and keep a track of how many elements of them are full, they are each going
                // to be num_samples in size at the most
                training_set_idx_t num_going_left = count_elements_matching(candidate_split_directions, LEFT);
                training_set_idx_t num_going_right = this->_num_training_samples - num_going_left;

                if ((num_going_left == 0) || (num_going_right == 0)) {
#ifdef VERBOSE                
                    std::cout << "found a split which doesn't separate data at all!" << std::endl;
#endif
                    continue;
                }

                // std::cout << "allocated left/right = " << candidate_samples_going_left << ", " << candidate_samples_going_right << std::endl;
                fill_sample_vectors(candidate_samples_going_left, candidate_samples_going_right,
                                    candidate_split_directions, this->_valid_indices);

#ifdef VERBOSE                                
                std::cout << num_going_left << " going left: ";
                print_vector_up_to<training_set_idx_t>(std::cout, candidate_samples_going_left, num_going_left);
                std::cout << num_going_right << " going right: ";
                print_vector_up_to<training_set_idx_t>(std::cout, candidate_samples_going_right, num_going_right);
#endif

                // use the training set to compute information gain (this could be unsupervised, 1d supervised, multi-d supervised etc etc)
                gain = this->_training_set.information_gain(_parent_dist, candidate_samples_going_left, candidate_samples_going_right,
                                                            _num_in_parent, num_going_left, num_going_right,
                                                            &candidate_left_dist, &candidate_right_dist);

                // NEW! Now we adjust the information gain depending on whether we want balanced splits or accuracy.
                // This stuff is down to Manik Varma, so CHECK what paper the stuff gets published in.
                // Need to cast all these to double as otherwise this seems to work purely with integers
                double balance_indicator = (abs(static_cast<double>(num_going_left) - static_cast<double>(num_going_right)) /
                                            static_cast<double>(_num_in_parent));
#ifdef VERBOSE
                std::cout << "final gain = " << entropy_adjuster << " * " << gain << " - "
                    << balance_adjuster << " * " << balance_indicator << " = "; // this output line concluded below
#endif

                gain *= entropy_adjuster;
                gain -= (balance_adjuster * balance_indicator);

#ifdef VERBOSE
                std::cout << gain << std::endl; // finishes the line begun above
                std::cout << "gain = " << gain << ", best gain so far = " << best_information_gain << std::endl;
#endif    

                if (gain >= best_information_gain) {
                    // if we are here, this is the best information gain we've found so far. Save all the data
                    best_information_gain = gain;
                    set_best_split_idx(i);
                    set_best_split_thresh(thresholds(i,j));
                    // _split_feature = dims_to_try(i);
                    // _split_value = thresholds(i,j);

                    found_a_good_gain = true;

                    // if we have previously stored a good set of left / right vectors, now we have found
                    // a better one, so need to delete the old ones
                    if (*samples_going_left != NULL) {
                        delete *samples_going_left;
                    }
                    if (*samples_going_right != NULL) {
                        delete *samples_going_right;
                    }

                    //allocate new output vectors (FIXME: can preallocate these too!)
                    *samples_going_left = new garf_types<training_set_idx_t>::vector(num_going_left);
                    *samples_going_right = new garf_types<training_set_idx_t>::vector(num_going_right);
                    {
                        // This might seem nuts but in the case of pyublas, indexing into vectors is quote unquote "much slower"
                        // than iterator access. Hence we do this.
                        garf_types<training_set_idx_t>::vector::iterator left_out_it = (*samples_going_left)->begin();
                        garf_types<training_set_idx_t>::vector::iterator candidate_left_it = candidate_samples_going_left.begin();
                        for (uint32_t i=0; i < num_going_left; i++) {
                            *left_out_it = *candidate_left_it;
                            left_out_it++;
                            candidate_left_it++;
                        }
                    }
                    {
                        garf_types<training_set_idx_t>::vector::iterator right_out_it = (*samples_going_right)->begin();
                        garf_types<training_set_idx_t>::vector::iterator candidate_right_it = candidate_samples_going_right.begin();
                        for (uint32_t i=0; i < num_going_right; i++) {
                            *right_out_it = *candidate_right_it;
                            right_out_it++;
                            candidate_right_it++;
                        }

                    }
                    // This is the best we've found so far, so let's put this data into the output variables
                    left_dist_out.fit_params(this->_training_set.dist_features(), **samples_going_left, num_going_left);
                    right_dist_out.fit_params(this->_training_set.dist_features(), **samples_going_right, num_going_right);
#ifdef VERBOSE
                    std::cout << "new best inf gain: " << gain
                        << " split index " << best_split_idx() << " on thresh " << best_split_thresh();
                    std::cout << " " << num_going_left << " going left, "
                        << num_going_right << " going right" << std::endl;
                    // print_vector_up_to<training_set_idx_t>(std::cout, **samples_going_left, num_going_left);
                    // print_vector_up_to<training_set_idx_t>(std::cout, **samples_going_right, num_going_right);
#endif
                }
            }
        }    

        if (!found_a_good_gain) {
            throw split_error();
        }
#ifdef VERBOSE
        std::cout << "final split found: "
            << (*samples_going_left)->size() << "/" 
            << (*samples_going_right)->size() << std::endl;
#endif
        return best_information_gain;
    }    

    // Once the subclasses have done their situation specific thing to calculate feature values, all the rest 
    // (calculating feature ranges, picking thresholds, assessing the information gain at each split) can
    // be taken care of generically.
    template<typename FeatT, typename LabelT>
    inf_gain_t split_finder<FeatT, LabelT>::pick_best_feature(const typename garf_types<FeatT>::matrix feature_values,
                                                        tset_indices_t** samples_going_left,
                                                        tset_indices_t** samples_going_right,
                                                        multivariate_normal<LabelT>& left_dist, 
                                                        multivariate_normal<LabelT>& right_dist,
                                                        boost::random::mt19937& gen) {
        // Calculate min and max for each feature, so we know what values to pick thresholds in between
        typename garf_types<FeatT>::vector feature_mins(this->_num_splits_to_try);
        typename garf_types<FeatT>::vector feature_maxs(this->_num_splits_to_try);
        this->calculate_mins_and_maxs(feature_values, feature_mins, feature_maxs);

#ifdef VERBOSE
        std::cout << "minx and maxs calculated" << std::endl;
        std::cout << "mins: " << feature_mins << std::endl;
        std::cout << "maxs: " << feature_maxs << std::endl;
#endif

        // Generate some thresholds in the above ranges
        typename garf_types<FeatT>::matrix thresholds(this->_num_splits_to_try, this->_num_threshes_per_split);
        this->generate_thresholds(feature_mins, feature_maxs, thresholds, gen);
#ifdef VERBOSE
        std::cout << "thresholds = " << thresholds << std::endl;
        std::cout << "calculating information gains" << std::endl;
#endif



        // Work out which of our candidate splits gives the best information gain.
        return this->calculate_information_gains(thresholds, feature_values,
                                                 samples_going_left, samples_going_right,
                                                 left_dist, right_dist);
    }
}


#endif