#ifndef GARF_REGRESSION_TREE_IMPL_HPP
#define GARF_REGRESSION_TREE_IMPL_HPP

namespace garf {

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_tree<FeatT, LabelT, SplitT>::init(const regression_forest<FeatT, LabelT, SplitT>* forest, tree_idx_t tree_id) {
        if (forest == NULL) {
            throw std::invalid_argument("NULL pointer for forest passed to regression_init::init()");
        }
        this->set_tree_id(tree_id);
        _forest = forest;
        _params = forest->get_params();
        _stats = forest->get_stats();
#ifdef VERBOSE        
        std::cout << "regression_tree::init() id " << this->tree_id() << /* " this = " << this << " rng == " << _rng.get() << */ std::endl;
#endif        
    }
    
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_tree<FeatT, LabelT, SplitT>::train(const supervised_training_set<FeatT, LabelT>& training_set) {
        _root.reset(new regression_node<FeatT, LabelT, SplitT>(NULL, this));

        // Need to construct a vector saying which indices have reached a node. Note this is a raw 'new'
        // allocation as opposed to any smart pointer jazz - this is because the root node will hold this
        // array in a shared_ptr, and therefore this array is automatically deallocated when the node is
        // deleted.
        training_set_idx_t num_training_samples = training_set.num_training_samples();
        tset_indices_t* samples_hitting_node = new tset_indices_t(num_training_samples);
           
        // Make a reference so the following code looks a bit nicer - indexing through a pointer
        // to a ublas array looks pretty crappy
        tset_indices_t & samples_hitting_node_ref = *samples_hitting_node;

        if (_params->_bagging) {
            // If we are doing bagging, then we the samples hitting node are
            // randomly resampled WITH REPLACEMENT from the whole dataset.
            uniform_int_distribution<> data_point_chooser(0, num_training_samples-1);
            for (training_set_idx_t i=0; i < num_training_samples; i++) {
                samples_hitting_node_ref(i) = data_point_chooser(*_rng);
            }
        } else {
            // Without bagging, just provide 0, 1, 2, .. n-1
            for (training_set_idx_t i=0; i < num_training_samples; i++) {
                samples_hitting_node_ref(i) = i;
            }        
        }
        
#ifdef VERBOSE
        std::cout << "bagging = " << _params->_bagging << ", giving tree " << this->tree_id() << " indices " << samples_hitting_node_ref << std::endl;
#endif

        multivariate_normal<LabelT>* root_distribution =
            new multivariate_normal<LabelT>(training_set.inf_gain_dimensionality());
        root_distribution->fit_params(training_set.dist_features(), *samples_hitting_node, num_training_samples);
        _root->train(training_set, samples_hitting_node, root_distribution, *_rng, _params);
    }   

    
    /* CRUCIAL! This now returns tree_idx_t instead of void. This refers to the node_id of the leaf node
      we stopped at, which in some cases is crucial for debugging */
    template<typename FeatT, typename LabelT, template< typename, typename> class SplitT>
    tree_idx_t regression_tree<FeatT, LabelT, SplitT>::predict(const matrix_row<const typename garf_types<FeatT>::matrix> & sample, 
                                                         typename garf_types<LabelT>::matrix::iterator2* prediction_mean,
                                                         typename garf_types<LabelT>::matrix::iterator2* prediction_var) const {
                                               
        // call the other predict() function with a max depth set to max_int or something
        return predict(sample, prediction_mean, prediction_var, std::numeric_limits<depth_t>::max());
    }


    template<typename FeatT, typename LabelT, template< typename, typename> class SplitT>
    node_idx_t regression_tree<FeatT, LabelT, SplitT>::predict(const matrix_row<const typename garf_types<FeatT>::matrix> & sample, 
                                                               typename garf_types<LabelT>::matrix::iterator2* prediction_mean, 
                                                               typename garf_types<LabelT>::matrix::iterator2* prediction_var,
                                                               depth_t max_depth) const {
                                                             
        // this may seem totally pointless, as I could have just passed in a reference in the first place. However
        // doing it this way means that at the place the function is called, the address-of operator makes it clear
        // that the argument is going to be modified.                                                     
        typename garf_types<LabelT>::matrix::iterator2 & prediction_mean_ref = *prediction_mean;
        typename garf_types<LabelT>::matrix::iterator2 & prediction_var_ref = *prediction_var;
        
        const regression_node<FeatT, LabelT, SplitT>* current_node = _root.get();
        
        // loop down through the nodes going left or right as needed
        // With the max depth thing, basically we limit the number of steps down we can go with the
        // while loop (above it is a 'while (true)') and therefore the thing might stop early. If we
        // limit the depth to zero then we won't go anywhere, the current node will just be
        depth_t depth_traversed = 0;
        while (depth_traversed < max_depth) {
            if (current_node->is_leaf()) {
                break; // reached the bottom of the tree
            }
            
            // If we haven't reached the bottom, look at the splitter and see which direction we should go in
            if (current_node->get_splitter().evaluate(sample) == LEFT) {
                current_node = current_node->left().get();
            }
            else {
                current_node = current_node->right().get();
            }
            depth_traversed++;
        }
        
        // Retrieve the Gaussian describing all the datapoints which landed at this point.
        const typename garf_types<LabelT>::vector & mean_at_leaf = current_node->get_sample_distribution().mu;
        const typename garf_types<LabelT>::matrix & covar_at_leaf = current_node->get_sample_distribution().sigma;
        uint32_t dims = mean_at_leaf.size();
        // FIXME: change this to use iterators
        for (uint32_t i=0; i < dims; i++) {
            prediction_mean_ref[i] = mean_at_leaf(i);
            prediction_var_ref[i] = covar_at_leaf(i, i);  // just go down diagonal and get regular variances..
        }
        return current_node->node_id();
    }
    

    template<typename FeatT, typename LabelT, template<typename, typename> class SplitT>
    node_idx_t regression_tree<FeatT, LabelT, SplitT>::max_depth() const {
        return _root->max_depth();
    }


    template<typename FeatT, typename LabelT, template< typename, typename> class SplitT>
    regression_node<FeatT, LabelT, SplitT> const& regression_tree<FeatT, LabelT, SplitT>::get_root() const {
        if (_root.get() == NULL) {
            throw std::logic_error("Root not present, tree must not be trained yet.");
        }
        return *_root;
    }

}



#endif