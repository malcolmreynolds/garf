#ifndef GARF_UNSUPERVISED_TRAINING_SET_IMPL_HPP
#define GARF_UNSUPERVISED_TRAINING_SET_IMPL_HPP

template<typename FeatT>
unsupervised_training_set<FeatT>::unsupervised_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix)
    : training_set<FeatT>(feature_matrix)
    {
    
}


// Given some indices into the training set pertaining to parent, left and right versions, compute the information gain
template<typename FeatT>
inf_gain_t unsupervised_training_set<FeatT>::information_gain(const multivariate_normal<FeatT> & parent_dist,
                                                              const tset_indices_t & left_indices,
                                                              const tset_indices_t & right_indices,
                                                              const training_set_idx_t num_in_parent,
                                                              const training_set_idx_t num_going_left,
                                                              const training_set_idx_t num_going_right,
                                                              multivariate_normal<FeatT> * temp_dist_l,
                                                              multivariate_normal<FeatT> * temp_dist_r) const {

    // Fit parameters to parent, left and right distributions
    temp_dist_l->fit_params(this->features(), left_indices, num_going_left);
    temp_dist_r->fit_params(this->features(), right_indices, num_going_right);
    
    return multivariate_normal<FeatT>::unsup_inf_gain(parent_dist, *temp_dist_l, *temp_dist_r,
                                                      num_in_parent, num_going_left, num_going_right);
}

#endif