#ifndef GARF_MULTI_SUPERVISED_REGRESSION_TRAINING_SET_IMPL_HPP
#define GARF_MULTI_SUPERVISED_REGRESSION_TRAINING_SET_IMPL_HPP

template<typename FeatT, typename LabelT>
multi_supervised_regression_training_set<FeatT,LabelT>::multi_supervised_regression_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix,
                                                                                                          boost::shared_ptr<typename garf_types<LabelT>::matrix> labels)
    : supervised_training_set<FeatT, LabelT>(feature_matrix),
      _labels(labels),
      _label_dimensionality(_labels->size2())
       {
    if (_labels->size1() != this->num_training_samples()) {
        throw std::invalid_argument("number of labels provided does not match number of training samples?");
    }
}           

// Given some indices into the training set pertainined to parent, left and right versions, compute the information gain
template<typename FeatT, typename LabelT>
inf_gain_t multi_supervised_regression_training_set<FeatT,LabelT>::information_gain(const multivariate_normal<LabelT> & parent_dist,
                                                                                    const tset_indices_t & left_indices,
                                                                                    const tset_indices_t & right_indices,
                                                                                    const training_set_idx_t num_in_parent,
                                                                                    const training_set_idx_t num_going_left,
                                                                                    const training_set_idx_t num_going_right,
                                                                                    multivariate_normal<FeatT> * temp_dist_l,
                                                                                    multivariate_normal<FeatT> * temp_dist_r) const {
    // Fit parameters to parent, left and right distributions
    temp_dist_l->fit_params(*_labels, left_indices, num_going_left);
    temp_dist_r->fit_params(*_labels, right_indices, num_going_right);
    
    // this is the 
    return multivariate_normal<FeatT>::unsup_inf_gain(parent_dist, *temp_dist_l, *temp_dist_r,
                                                      num_in_parent, num_going_left, num_going_right);
}


#endif