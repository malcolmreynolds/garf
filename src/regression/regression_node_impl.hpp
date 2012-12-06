#ifndef GARF_REGRESSION_NODE_IMPL_HPP
#define GARF_REGRESSION_NODE_IMPL_HPP

namespace garf {
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    regression_node<FeatT, LabelT, SplitT>::regression_node(const regression_node<FeatT, LabelT, SplitT>* const parent,
                                                            regression_tree<FeatT, LabelT, SplitT>* const tree)
        : decision_node(parent == NULL ? 0 : parent->depth()+ 1), 
          _parent(parent), _tree(tree) {
        this->set_node_id(_tree->get_next_node_id());
#ifdef VERBOSE
        std::cout << "regression_node() id " << _tree->tree_id() << ":"<< this->node_id() << " depth " << this->depth() << std::endl;
#endif  
    }
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    regression_node<FeatT, LabelT, SplitT>::~regression_node() {
    #ifdef VERBOSE
        std::cout << "~regression_node id " << _tree->tree_id() << ":" << this->node_id() << std::endl;
    #endif  
    };

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_node<FeatT, LabelT, SplitT>::train(const supervised_training_set<FeatT, LabelT> & training_set,
                                                       const garf_types<training_set_idx_t>::vector* samples_landing_here,
                                                       const multivariate_normal<LabelT>* sample_distribution,
                                                       boost::random::mt19937& rng,
                                                       const regression_forest_params* const params) {
        // Store everything this node needs to identify the data landing here (note that all this stuff
        // was computed by the parent node's split function)
        this->set_samples_landing_here(samples_landing_here);
        if (this->num_samples_landing_here() == 0) {
            throw std::logic_error("We should never be trying to train a node with zero datapoints.");
        }
        if (sample_distribution == NULL) {
            throw std::logic_error("sample_distribution is NULL in regression_node::train");
        }
        _sample_distribution.reset(sample_distribution);
        
        // Everything is set up at this node. Now let's see if we should split further
        if (this->termination_conditions_met(params)) {
            return;
        }

        // Also do a second test, based on the sample distribution
        if (variance_below_minimum(params)) {
            std::cout << "node #" << _tree->tree_id() << ":" << this->node_id()
                << " stopping at depth " << this->depth() << " with " << this->num_samples_landing_here()
                << " samples due to variance " << _sample_distribution->sigma << " < " 
                << params->_min_variance << std::endl;
            return;
        }
        
        // Create whatever kind of splitter the template indicates
        _splitter.reset( new SplitT<FeatT, LabelT>(training_set, *samples_landing_here,
                                                   *_sample_distribution, this->num_samples_landing_here(),
                                                   params->_num_splits_to_try, params->_num_threshes_per_split));

        // Set these parameters - needed for Manik Varma's splitting thing
        // Say depth is plus one as we will be raising to this power, and if it is zero then at the root node
        // raising to the power zero will mean the optimisation only pays attention to whether the split is balanced,
        // and not at all to the entropy on each side
        _splitter->set_depth(this->depth() + 1); 
        _splitter->set_balance_bias(params->_balance_bias);
            
        // create variables to hold left / right split information - these will be filled in by the splitter
        multivariate_normal<FeatT>* left_child_dist =
            new multivariate_normal<LabelT>(training_set.inf_gain_dimensionality());
        multivariate_normal<FeatT>* right_child_dist = 
            new multivariate_normal<LabelT>(training_set.inf_gain_dimensionality());
        garf_types<split_direction_t>::vector split_directions(this->num_samples_landing_here());
        // we don't know the size of these vectors yet, so pass in addresses to these pointers and find_optimal_split 
        // will set them to vectors of the right size
        garf_types<training_set_idx_t>::vector* samples_going_left = NULL;
        garf_types<training_set_idx_t>::vector* samples_going_right = NULL;

        try {
            inf_gain_t information_gain = _splitter->find_optimal_split(&samples_going_left, &samples_going_right,
                                                                        *left_child_dist, *right_child_dist, rng);
            // Store the information gain captured so we can access this and examine via python.
            this->set_information_gain(information_gain);
        }
        catch(split_error &error) {
#ifdef USING_OPENMP          
            std::cout << (threadsafe_ostream()
                          << "Tree " << _tree->tree_id() 
                          << " stopped growing at depth " 
                          << this->depth() 
//                          << " : " << error.what() 
                          << '\n').toString(); 
#else
            std::cout << "Tree " << _tree->tree_id()
                << " stopped growing at depth " << this->depth()
                << " with " << this->num_samples_landing_here() << " samples : "
                << error.what() << std::endl;
#endif                    
            return;
        }    
      
        // New test! If the best split we can come up with doesn't have > min samples in BOTH
        // branches, then we assume we have started overfitting, and bomb out early
        int num_going_left = samples_going_left->size();
        int num_going_right = samples_going_right->size();

        if ((num_going_left < params->_min_sample_count) ||
            (num_going_right < params->_min_sample_count)) {
            std::cout << "regression_node::train() id = (" << _tree->tree_id() << "," << this->node_id()
                << "): stopping overfitting, best split would have been (" << num_going_left << "/"
                << num_going_right << ")" << std::endl;
            return;
        }

        std::cout << "regression_node::train() id = (" << _tree->tree_id() << "," << this->node_id() 
            << "): " << this->num_samples_landing_here() << " -> " << num_going_left << " / "
            << num_going_right << std::endl;

        // Create the two children and train recursively
        _left.reset(new regression_node<FeatT, LabelT, SplitT>(this, _tree));
        _left->train(training_set, samples_going_left, left_child_dist, rng, params);

        _right.reset(new regression_node<FeatT, LabelT, SplitT>(this, _tree));
        _right->train(training_set, samples_going_right, right_child_dist, rng, params);  
        
    }

    
    template<typename FeatT, typename LabelT, template<typename, typename> class SplitT>
    node_idx_t regression_node<FeatT, LabelT, SplitT>::max_depth() const {
        if (is_leaf()) {
            return depth(); // or should this be zero?
        }
        // Our depth is, by definition, one more than the biggest depth of left and right
        node_idx_t left_max_depth = _left->max_depth();
        node_idx_t right_max_depth = _right->max_depth();

        if (left_max_depth > right_max_depth) {
            return left_max_depth;
        }
        else {
            return right_max_depth;
        }
    }
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    inline regression_node<FeatT, LabelT, SplitT> const& regression_node<FeatT, LabelT, SplitT>::get_left() const {
        if (_left.get() == NULL) {
            throw std::logic_error("Node doesn't have a left child");
        }
        return *_left;
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    inline regression_node<FeatT, LabelT, SplitT> const& regression_node<FeatT, LabelT, SplitT>::get_right() const {
        if (_right.get() == NULL) {
            throw std::logic_error("Node doesn't have a right child");
        }
        return *_right;
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    inline SplitT<FeatT, LabelT> const& regression_node<FeatT, LabelT, SplitT>::get_splitter() const {
        if (_splitter.get() == NULL) {
            throw std::logic_error("Node doesn't have a split");
        }
        return *_splitter;
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    bool regression_node<FeatT, LabelT, SplitT>::variance_below_minimum(const regression_forest_params* const params) const {
        // If we have any features with variance below this threshold, then time to 
        // stop training even if we are way below the max depth / min sample count.
        LabelT min_variance = params->_min_variance;

        if (min_variance <= 0.0) {
            // std::cout << "min_variance parameter not set, so returning early" << std::endl;
            return false;
        }

        uint32_t output_feature_dimensionality = _sample_distribution->sigma.size1();
        for (uint32_t i = 0; i < output_feature_dimensionality; i++) {
            // Only check the variances, ie the diagonals on sigma. If any of the variances
            // are high, then we keep going (ie there could be zero variance in one dimension
            // but high variance in another)
            if (_sample_distribution->sigma(i, i) > min_variance) {
                return false; // at least this variance is high enough to carry on splitting
            }
        }

        // If we are here, then all the variances must be low, so we stop
        return true;
    }
}


#endif