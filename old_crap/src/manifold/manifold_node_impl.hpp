#ifndef GARF_MANIFOLD_NODE_IMPL_HPP
#define GARF_MANIFOLD_NODE_IMPL_HPP

/* Store the parent node, forest and parameters in preparation for training */
template<typename FeatT>
manifold_node<FeatT>::manifold_node(const manifold_node<FeatT>* const parent,
                                          manifold_tree<FeatT>* const tree) : 
      // parent being null means this is the root node, so depth zero
      decision_node(parent == NULL ? 0 : parent->depth()+ 1), 
      _parent(parent), _tree(tree) {
    this->set_node_id(_tree->get_next_node_id());
#ifdef VERBOSE
    std::cout << "manifold_node() id " << _tree->tree_id() << ":"<< this->node_id() << " depth " << this->depth() << std::endl;
#endif
}

template<typename FeatT>
manifold_node<FeatT>::~manifold_node() {
#ifdef VERBOSE
    std::cout << "~manifold_node id " << _tree->tree_id() << ":" << this->node_id() << std::endl;
#endif  
};



template<typename FeatT>
void manifold_node<FeatT>::train(training_set<FeatT>& training_set,
                                       tset_indices_t* samples_landing_here,
                                       multivariate_normal<FeatT>* sample_distribution,
                                       boost::random::mt19937& rng,
                                       const manifold_forest_params* const params) {

    this->set_samples_landing_here(samples_landing_here);
    if (this->num_samples_landing_here() == 0) {
        throw std::logic_error("We should never be trying to train a node with zero data points");
    }
    if (sample_distribution == NULL) {
        throw std::invalid_argument("sample_distribution == NULL");
    }
    _sample_distribution.reset(sample_distribution);
#ifdef VERBOSE
    std::cout << "manifold_node::train() tree_id = " << _tree->tree_id() <<" node_id = " << this->node_id()
              << " depth = " << this->depth()
              // << " samples_landing_here = " << *samples_landing_here
              << " num_samples_landing_here = " << this->num_samples_landing_here()
              << " dist.mu = " << _sample_distribution->mu 
              << " dist.sigma = " << _sample_distribution->sigma << std::endl;
#endif
    // Test whether we have hit max depth, etc
    if (this->termination_conditions_met(params)) {
        return;
    }

    // If we are here then we need to pick a bunch of candidate splits, evaluate which one is the best,
    // then use that to pick what goes left and what goes right. First we decide based on the forest params
    // what kind of splitter to generate (in future some entries for the split_type value could do a probabilistic
    // decision.)
    switch(params->_split_type) {
    case AXIS_ALIGNED:
        _splitter.reset(new axis_aligned_split_finder<FeatT, FeatT>(training_set, *samples_landing_here,
                                                                    *_sample_distribution, this->num_samples_landing_here(),
                                                                    params->_num_splits_to_try,
                                                                    params->_num_threshes_per_split));
        break;
    default:
        throw std::invalid_argument("only Axis aligned splits supported for now");
    }
    
    // variables to hold left / right directions 
    multivariate_normal<FeatT>* left_child_dist =
        new multivariate_normal<FeatT>(training_set.inf_gain_dimensionality());
    multivariate_normal<FeatT>* right_child_dist = 
        new multivariate_normal<FeatT>(training_set.inf_gain_dimensionality());
    garf_types<split_direction_t>::vector split_directions(this->num_samples_landing_here());
    // we don't know the size of these vectors yet, so pass in addresses to these pointers and find_optimal_split 
    // will set them to vectors of the right size
    garf_types<training_set_idx_t>::vector* samples_going_left = NULL;
    garf_types<training_set_idx_t>::vector* samples_going_right = NULL;
    
    try {
        _splitter->find_optimal_split(&samples_going_left, &samples_going_right, *left_child_dist, *right_child_dist, rng);
    }
    catch(split_error &error) {
        std::cout << "Caught a split_error, stopping growing here: "<< error.what() << std::endl;
        return;
    }

#ifdef VERBOSE
    std::cout << "manifold_node::train() id = (" << _tree->tree_id() << "," << this->node_id() 
        << "): split of " << this->num_samples_landing_here() << " datapoints found with " << samples_going_left->size() << " going left and "
        << samples_going_right->size() << " going right." << std::endl;
#endif
    

    // Create the two children and train recursively
    _left.reset(new manifold_node<FeatT>(this, _tree));
    _left->train(training_set, samples_going_left, left_child_dist, rng, params);
    
    _right.reset(new manifold_node<FeatT>(this, _tree));
    _right->train(training_set, samples_going_right, right_child_dist, rng, params);
}

#endif