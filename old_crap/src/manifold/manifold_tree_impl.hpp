#ifndef GARF_MANIFOLD_TREE_IMPL_HPP
#define GARF_MANIFOLD_TREE_IMPL_HPP

template<typename FeatureType>
void manifold_tree<FeatureType>::init(const manifold_forest<FeatureType>* forest, tree_idx_t tree_id) {
    if (forest == NULL) {
        throw std::invalid_argument("NULL pointer passed to manifold_init::init()");
    }
    this->set_tree_id(tree_id);
    _forest = forest;
    _params = forest->get_params();
    _stats = forest->get_stats();
    std::cout << "manifold_tree::init() id " << this->tree_id() << std::endl;
}

template<typename FeatureType>
void manifold_tree<FeatureType>::train(training_set<FeatureType>& training_set) {
    _root.reset(new manifold_node<FeatureType>(NULL, this));

    // Need to construct a vector saying which indices have reached a node.
    training_set_idx_t num_training_samples = training_set.num_training_samples();
    garf_types<training_set_idx_t>::vector* samples_hitting_node = 
       new garf_types<training_set_idx_t>::vector(num_training_samples);

    if (_params->_bagging) {
        // If we are doing bagging, then we the samples hitting node are
        // randomly resampled WITH REPLACEMENT from the whole dataset.
        uniform_int_distribution<> data_point_chooser(0, num_training_samples-1);
        for (training_set_idx_t i=0; i < num_training_samples; i++) {
            (*samples_hitting_node)[i] = data_point_chooser(*_rng);
        }
    } else {
        // Without bagging, just provide 0, 1, 2, .. n-1
        for (training_set_idx_t i=0; i < num_training_samples; i++) {
            (*samples_hitting_node)[i] = i;
        }        
    }
    
    multivariate_normal<FeatureType>* root_distribution =
        new multivariate_normal<FeatureType>(training_set.inf_gain_dimensionality());
    root_distribution->fit_params(training_set.dist_features(), *samples_hitting_node, num_training_samples);
    _root->train(training_set, samples_hitting_node, root_distribution, *_rng, _params);

}



// computes the affinity matrix for a given tree, returns a const reference to it
template<typename FeatureType>
const garf_types<double>::matrix& manifold_tree<FeatureType>::compute_tree_affinity_matrix() {
    std::cout << "calculating affinity in tree " << this->tree_id() << std::endl;
    
    // allocate a new affinity matrix
    _affinity_matrix.reset(new garf_types<double>::matrix(_stats->_num_training_samples, _stats->_num_training_samples));
    _affinity_matrix->clear();
    
    // need to visit every leaf node and look at the 'samples_landing_here' field, then put some corresponding
    // affinity into the matrix. We use a stack to store pointers to nodes we need to visit 
    std::stack<boost::shared_ptr<manifold_node<FeatureType> > > need_to_visit;
    need_to_visit.push(_root);
    
    while (!need_to_visit.empty()) {
        boost::shared_ptr<manifold_node<FeatureType> > n = need_to_visit.top();
        need_to_visit.pop();
        if (n.get() == NULL) {
            throw std::logic_error("NULL node on stack");
        }
        if (n->is_leaf()) {
            compute_leaf_affinity(*n);
        }
        else {
            need_to_visit.push(n->left());
            need_to_visit.push(n->right());
        }    
    }
    return *_affinity_matrix;
}

// For a single leaf node, put the elements into the affinity matrix
template<typename FeatureType>
void manifold_tree<FeatureType>::compute_leaf_affinity(const manifold_node<FeatureType>& node) {
    garf_types<double>::matrix& affinity_matrix_ref = *_affinity_matrix;
    const tset_indices_t& samples_landing_here = node.samples_landing_here();
    training_set_idx_t num_samples_landing_here = samples_landing_here.size();
    
    // Gaussian parameter - store this somewhere sensible
    double sigma_2 = 1;
    // Need to do a dot product between two data points
    double dot_product_result;
    const typename garf_types<FeatureType>::matrix & features = _forest->get_training_set().features();

    switch(_params->_affinity_distance_type) {
    case BINARY:
        for (training_set_idx_t i=0; i < num_samples_landing_here; i++) {
            training_set_idx_t data_idx_i = samples_landing_here(i);
            
            // We don't set any affinity elements on the diagonal!
            // affinity_matrix_ref(data_idx_i, data_idx_i) = 1.0; //binary affinity here
            for (training_set_idx_t j=i+1; j < num_samples_landing_here; j++) {
                training_set_idx_t data_idx_j = samples_landing_here(j);
                affinity_matrix_ref(data_idx_i, data_idx_j) = 1.0;
                affinity_matrix_ref(data_idx_j, data_idx_i) = 1.0;
            }
        }
        break;
    case GAUSSIAN:
        for (training_set_idx_t i=0; i < num_samples_landing_here; i++) {
            training_set_idx_t data_idx_i = samples_landing_here(i);
            
            for (training_set_idx_t j=(i+1); j < num_samples_landing_here; j++) {
                training_set_idx_t data_idx_j = samples_landing_here(j);

                dot_product_result = 0;
                for (uint32_t k=0; k < features.size2(); k++) {
                    dot_product_result += (features(data_idx_i, k) * features(data_idx_j, k));
                }
                dot_product_result /= sigma_2;
                affinity_matrix_ref(data_idx_i, data_idx_j) = dot_product_result;
                affinity_matrix_ref(data_idx_j, data_idx_i) = dot_product_result;
            }
        }
        break;
        
    default:
        throw std::logic_error("distance metric not supported");
    }
}

#ifdef BUILD_PYTHON_BINDINGS
template<typename FeatureType>
manifold_node<FeatureType> const& manifold_tree<FeatureType>::get_root() const {
    if (_root.get() == NULL) {
        throw std::logic_error("Root not present, tree must not be trained yet.");
    }
    return *_root;
}


#endif

#endif