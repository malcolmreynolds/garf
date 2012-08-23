#ifndef GARF_MANIFOLD_FOREST_IMPL_HPP
#define GARF_MANIFOLD_FOREST_IMPL_HPP

// this file should only ever be included by manifold_forest.hpp. I've just separated this
// stuff out so the header file only has declarations, and is somewhat easy to skim through

template<typename FeatureType>
manifold_forest<FeatureType>::manifold_forest() : _params(), _stats() {
}

template<typename FeatureType>
manifold_forest<FeatureType>::manifold_forest(manifold_forest_params params) :
 _params(params), _stats() {
     
}


template<typename FeatureType>
void manifold_forest<FeatureType>::hello() {
    std::cout << "Hello from manifold_forest!" << std::endl;
}

/* train_ is used so we can  */
template<typename FeatureType>
void manifold_forest<FeatureType>::train(boost::shared_ptr<training_set<FeatureType> > training_set) {
//        std::cout << "Received training matrix of dimensionality " << features->size1() << "," << features->size2() << std::endl;

    // Store the training set - note this could be one of a number of types!
    _training_set = training_set;

    _stats._num_training_samples = training_set->num_training_samples();
    _stats._feature_dimensionality = training_set->feature_dimensionality();

    std::cout << "manifold_forest::train received " << _stats._num_training_samples
        << " samples of dimensionality " << _stats._feature_dimensionality << ":" << std::endl;
#ifdef VERBOSE
    for (uint32_t i=0; i < _stats._num_training_samples; i++) {
        for (uint32_t dim_idx=0; dim_idx < _stats._feature_dimensionality; dim_idx++) {
            std::cout << _training_set->features()(i, dim_idx) << " ";
        }
        std::cout << std::endl;
    }
#endif

    _trees.reset(new manifold_tree<FeatureType>[_params._max_num_trees]);
    _stats._num_trees = 0;
    for (tree_idx_t i=0; i < _params._max_num_trees; i++) {
        std::cout << "training tree " << i << std::endl;
        _trees[i].init(this, i);
        _trees[i].train(*_training_set);
        _stats._num_trees++;
    }
}

#ifdef BUILD_PYTHON_BINDINGS
template<typename FeatureType>
void manifold_forest<FeatureType>::train_py(pyublas::numpy_matrix<FeatureType> feature_matrix) {
    // FIXME: this is doing a copy for now. Can we / SHOULD we fix it to operate otherwise? If we 
    // try and make it not copy, might it be crashable from python side?
    
    // Allocate a new matrix that python doesn't know about, copy the data
    boost::shared_ptr<typename garf_types<FeatureType>::matrix > _feature_matrix(new typename garf_types<FeatureType>::matrix(feature_matrix.size1(), feature_matrix.size2()));
    _feature_matrix->assign(feature_matrix);
    
    //allocate the training set - unsupervised for now
    boost::shared_ptr<training_set<FeatureType> > _training_set(new unsupervised_training_set<FeatureType>(_feature_matrix));
    
    train(_training_set);
}

template<typename FeatureType>
void manifold_forest<FeatureType>::train_sup_py(pyublas::numpy_matrix<FeatureType> feature_matrix, pyublas::numpy_matrix<double> label_matrix) {
    // If we have a different number of entries in features and labels, we have a problem
    if (feature_matrix.size1() != label_matrix.size1()) {
        throw std::invalid_argument("We must have the same number of features and labels");
    }
    
    //Allocate new matrices in shared_ptrs for both the feature matrix and label matrix
    boost::shared_ptr<typename garf_types<FeatureType>::matrix> _feature_matrix(new typename garf_types<FeatureType>::matrix(feature_matrix.size1(), feature_matrix.size2()));
    _feature_matrix->assign(feature_matrix);
    boost::shared_ptr<garf_types<double>::matrix> _label_matrix(new garf_types<double>::matrix(label_matrix.size1(), label_matrix.size2()));
    _label_matrix->assign(label_matrix);
    
    // allocate the training set
    boost::shared_ptr<training_set<FeatureType> > _training_set(new multi_supervised_regression_training_set<FeatureType, double>(_feature_matrix, _label_matrix));
    
    std::cout << "received " << _training_set->num_training_samples() << " samples with " 
        << _training_set->feature_dimensionality() << " feature dimensions and "
        << _label_matrix->size2() << " label dimensions." << std::endl;
    
    train(_training_set);
}

template<typename FeatureType>
pyublas::numpy_matrix<double> manifold_forest<FeatureType>::get_affinity_matrix_py() {
    // Generate the affinity matrix, store an internal forest copy
    if (_affinity_matrix.get() == NULL) {
        throw std::logic_error("Affinity matrix doesn't seem to have been calculated yet");
    }
    pyublas::numpy_matrix<double>* ret_mtx = new pyublas::numpy_matrix<double>(_affinity_matrix->size1(), _affinity_matrix->size2());
    ret_mtx->assign(*_affinity_matrix);
    return *ret_mtx;
}

template<typename FeatureType>
manifold_tree<FeatureType> const& manifold_forest<FeatureType>::get_tree(tree_idx_t idx) const {
    if (idx < _stats._num_trees) {
        return _trees[idx];
    }
    throw std::invalid_argument("index of tree required is not within range");
}

//template<typename FeatureType>


#endif

// Get the dxd affinity matrix from a forest, where a d-element training set was provided
template<typename FeatureType>
void manifold_forest<FeatureType>::compute_affinity_matrix() {

    _affinity_matrix.reset(new garf_types<double>::matrix(_stats._num_training_samples, _stats._num_training_samples));
    _affinity_matrix->clear();

    garf_types<double>::matrix& af_mtx_ref = *_affinity_matrix;

    for (uint32_t i=0; i < _stats._num_trees; i++) {
        af_mtx_ref += _trees[i].compute_tree_affinity_matrix();
#ifdef VERBOSE
        std::cout << "done tree " << i << std::endl;
        std::cout << "sum total = " << af_mtx_ref << std::endl;
#endif
    }
    
    // divide by number of trees we have
    af_mtx_ref /= _stats._num_trees;
#ifdef VERBOSE
    std::cout << "affinity after averaging = " << af_mtx_ref << std::endl;
#endif
}



#endif