#ifndef GARF_REGRESSION_FOREST_IMPL_HPP
#define GARF_REGRESSION_FOREST_IMPL_HPP

namespace garf {


#ifdef USING_TBB
    // This is the helper class that we rely on if Thread Buildin Blocks is 
    // enabled. No point compiling it otherwise, hence ifdef guard.

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    class concurrent_tree_trainer {
        // This class just contains pointers to everything we will need to access 
        // in order to train a tree
        const regression_forest<FeatT, LabelT, SplitT> * _forest;
        boost::shared_array<regression_tree<FeatT, LabelT, SplitT> > _trees;
        boost::shared_ptr<const supervised_training_set<FeatT, LabelT> > _training_set;
        atomic<tree_idx_t> & _num_trees;
    public:
        // Does the training of a tree for whatever range TBB provides
        void operator() (const blocked_range<training_set_idx_t> & r) const {
            for (training_set_idx_t i = r.begin(); i != r.end(); i++) {
#ifdef VERBOSE
                std::cout << (threadsafe_ostream() << " training tree " << i << " _num_trees = " << _num_trees << '\n').toString();
#endif                
                _trees[i].init(_forest, i);
                _trees[i].train(*_training_set);

                _num_trees.fetch_and_add(1);
            }
        }
        // Constructor passes in all the crucial bits
        concurrent_tree_trainer(const regression_forest<FeatT, LabelT, SplitT> * forest,
                                boost::shared_array<regression_tree<FeatT, LabelT, SplitT> > trees,
                                boost::shared_ptr<const supervised_training_set<FeatT, LabelT> > training_set,
                                atomic<tree_idx_t> & num_trees)
          : _forest(forest), _trees(trees), _training_set(training_set), _num_trees(num_trees) { 
        }
    };


#endif


    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    regression_forest<FeatT, LabelT, SplitT>::regression_forest()
        : _params(), _stats() {
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    regression_forest<FeatT, LabelT, SplitT>::regression_forest(regression_forest_params params)
        : _params(params), _stats() {
    }    
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::train(boost::shared_ptr<const supervised_training_set<FeatT, LabelT> > training_set) {
        _training_set = training_set;

        _stats._num_training_samples = training_set->num_training_samples();
        _stats._feature_dimensionality = training_set->feature_dimensionality();
        _stats._label_dimensionality = training_set->label_dimensionality();

        // std::cout << "regression_forest::train received " << _stats._num_training_samples
        //     << " samples of dimensionality " << _stats._feature_dimensionality << ":" << std::endl;
    #ifdef VERBOSE
        for (uint32_t i=0; i < _stats._num_training_samples; i++) {
            for (uint32_t dim_idx=0; dim_idx < _stats._feature_dimensionality; dim_idx++) {
                std::cout << _training_set->features()(i, dim_idx) << " ";
            }
            std::cout << std::endl;
        }
    #endif

        _trees.reset(new regression_tree<FeatT, LabelT, SplitT>[_params._max_num_trees]);
        _stats._num_trees = 0;

        // Keep track of the time
        time_t start, end;
        time(&start);

#ifdef USING_OPENMP 
        #pragma omp parallel for
        for (tree_idx_t i=0; i < _params._max_num_trees; i++) {
#ifdef VERBOSE            
            std::cout << (threadsafe_ostream() << "thread " << omp_get_thread_num()
                          << " training tree " << i << '\n').toString();
#endif                          

            _trees[i].init(this, i);
            _trees[i].train(*_training_set);
            // Need to make sure the load/store that makes up this increment is atomic
            // else we might undercount the number of trees once we are done.
            #pragma omp atomic
            _stats._num_trees++;
        }
#elif USING_TBB // Intel Thread Building Blocks
        atomic<tree_idx_t> tree_counter;
        tree_counter = 0;
        parallel_for(blocked_range<tree_idx_t>(0, _params._max_num_trees),
                     concurrent_tree_trainer<FeatT, LabelT, SplitT>(this,
                                                                    _trees,
                                                                    _training_set,
                                                                    tree_counter));
        _stats._num_trees = tree_counter;                                                                    
#else // not using OpenMP

        for (tree_idx_t i=0; i < _params._max_num_trees; i++) {
            _trees[i].init(this, i);
            _trees[i].train(*_training_set);
            _stats._num_trees++;
        }
#endif 
        // Check if any of the trees have no splits
        for (tree_idx_t i=0; i < _stats._num_trees; i++) {
            if (_trees[i].get_root().is_leaf()) {
                std::cout << "WARNING! Tree " << i << " didn't end up making any splits"
                    << " - most likely this means your stopping criteria are screwed. "
                    << "I advise you to reconsider your current path through life." << std::endl;
            }
        }

        // Training done, print out time
        time(&end);
        this->set_training_time(difftime(end, start));
        std::cout << "Trained " << _stats._num_trees << " trees";
        if (_trees[0].get_root().is_internal()) {  //make sure there is a split to look at
            std::cout << " with " << _trees[0].get_root().get_splitter().name();
        }
        std::cout << " in " << ((double)this->training_time()) << "s" << std::endl;
        std::cout << "number of nodes per tree: ";
        // Also calculate mean number of nodes
        uint32_t num_nodes;
        uint32_t num_nodes_accum = 0;
        for (uint32_t i = 0; i < _stats._num_trees; i++) {
            num_nodes = _trees[i].next_free_node_id();
            num_nodes_accum += num_nodes;
            std::cout << num_nodes << " ";
        }
        std::cout << std::endl << "mean nodes per tree: "
            << (num_nodes_accum / (double)_stats._num_trees) << std::endl;;


    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>    
    void regression_forest<FeatT, LabelT, SplitT>::prepare_tree_indices(typename garf_types<tree_idx_t>::vector trees_to_use,
                                                                        tree_idx_t num_trees_to_use,
                                                                        bool random_tree_permutations) const {
        // First fill up the array with just the indices of all trees
        for (tree_idx_t i = 0; i < _stats._num_trees; i++) {
            trees_to_use(i) = i;
        }

        // Now do a Knuth shuffle, although we stop before the end if we are only doing a few trees as it doesnt
        // seem to matter
        if (random_tree_permutations) {
            knuth_shuffle<tree_idx_t>(trees_to_use, num_trees_to_use, _trees[0]._rng.get());
        }

        std::cout << "predicting with the first " << num_trees_to_use << " of: " << trees_to_use << std::endl;         
    }

    /* This does all the prediction for a ful dataset. */
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::predict(typename garf_types<FeatT>::matrix const & feature_matrix,
                                                           typename garf_types<LabelT>::matrix* predictions,
                                                           typename garf_types<LabelT>::matrix* variances, 
                                                           typename garf_types<node_idx_t>::matrix* leaf_node_indices,
                                                           tree_idx_t num_trees_to_use,
                                                           depth_t max_depth,
                                                           bool use_weighted_average) {
        if (predictions == NULL) { throw std::runtime_error("null pointer passed in for predictions output array"); }
        if (variances == NULL) { throw std::runtime_error("null pointer passed in for variance output array"); }
        typename garf_types<LabelT>::matrix & predictions_out_ref = *predictions;
        typename garf_types<LabelT>::matrix & variances_out_ref = *variances;
        
        bool random_tree_permutations = true;
        // Zero trees is obviously useless - this is the shortcut argument to say we should use them all
        if (num_trees_to_use == 0) {
            num_trees_to_use = _stats._num_trees;
            random_tree_permutations = false; // no point randomly shuffling all the trees
        }
        else if (num_trees_to_use == _stats._num_trees) {
            random_tree_permutations = false; // no point randomly shuffling all
        }
        else if (num_trees_to_use > _stats._num_trees) {
            throw std::runtime_error("Was asked to predict on more trees than we actually have");
        }
        
        // As above with variance, work out whether we need to output leaf node indices
        bool output_leaf_nodes = (leaf_node_indices != NULL);

        if (output_leaf_nodes) {
            if (random_tree_permutations) {
                throw std::runtime_error("output_leaf_nodes and random_tree_permutations is a silly combination");
            }
            if (leaf_node_indices->size1() != feature_matrix.size1()) {
                throw std::runtime_error("leaf_node_indices->size1() should match number of items to predict on");
            }
            if (leaf_node_indices->size2() != num_trees_to_use) {
                throw std::runtime_error("leaf_node_indices->size2() should match number of trees we are predicting with");
            }
        }

        const training_set_idx_t num_samples_to_predict = feature_matrix.size1();
        const feature_idx_t dimensionality = feature_matrix.size2();
        
        const feature_idx_t output_dimensions = predictions->size2();
        
        if (output_dimensions != _training_set->inf_gain_dimensionality()) {
            throw std::runtime_error("predictions array has a size which doesn't match the dimensionality the forest was trained on!");
        }
        
        std::cout << "using " << num_trees_to_use << " trees to predict on " << num_samples_to_predict 
            << " " << dimensionality << "-dimensional samples creating " 
            << output_dimensions << "-d output" << std::endl;
        
        // For the moment keep it simple - Store the prediction of every forest in here
        typename garf_types<LabelT>::matrix outputs_per_tree_mean(num_trees_to_use, output_dimensions);
        typename garf_types<LabelT>::matrix outputs_per_tree_var(num_trees_to_use, output_dimensions);

        typename garf_types<LabelT>::matrix::iterator1 outputs_mean_row_itr;
        typename garf_types<LabelT>::matrix::iterator2 outputs_mean_col_itr;
        typename garf_types<LabelT>::matrix::iterator1 outputs_var_row_itr;
        typename garf_types<LabelT>::matrix::iterator2 outputs_var_col_itr;

        // If we are only predicting on some number n (< _stats._num_trees) of the trees, then we can either use
        // the first n trees or do a random permutation. Choose here.
        // Note that when we use all the trees this is kind of a waste of effort, BUT
        // the programming to do it all super-efficiently would be messy. Realistically, generating a 1000 element
        // vector of 0 .. 999 is a totally neglibile cost.
        garf_types<tree_idx_t>::vector trees_to_use(_stats._num_trees);
        prepare_tree_indices(trees_to_use, num_trees_to_use, random_tree_permutations);

        // loop over data items
        for (training_set_idx_t pred_idx = 0; pred_idx < num_samples_to_predict; pred_idx++) {
            const matrix_row<const typename garf_types<FeatT>::matrix> current_sample(feature_matrix, pred_idx);

#ifdef VERBOSE
            std::cout << "predicting on sample " << pred_idx << ": " << current_sample << std::endl;
#endif
            
            node_idx_t final_node_index;
            // loop over trees, get an iterator to each row of the output matrix in turn, then give
            // to the multivariate normal so that the mean variance can be calculated
            outputs_mean_row_itr = outputs_per_tree_mean.begin1();
            outputs_var_row_itr = outputs_per_tree_var.begin1();

            for (tree_idx_t t=0; t < num_trees_to_use; t++) {
                outputs_mean_col_itr = outputs_mean_row_itr.begin();
                outputs_var_col_itr = outputs_var_row_itr.begin();
                final_node_index = _trees[trees_to_use(t)].predict(current_sample, &outputs_mean_col_itr, 
                                                                   &outputs_var_col_itr, max_depth);

                if (output_leaf_nodes) {
                    (*leaf_node_indices)(pred_idx, t) = final_node_index;
                }

                outputs_mean_row_itr++;
                outputs_var_row_itr++;
            }

            // Combine the results across all trees in the desired way
            if (use_weighted_average) {
                weighted_average(predictions_out_ref, variances_out_ref, pred_idx,
                                 outputs_per_tree_mean, outputs_per_tree_var);
            }
            else {
                unweighted_average(predictions_out_ref, variances_out_ref, 
                                   pred_idx, outputs_per_tree_mean); // don't need variances as this average is unweighted!
            }
        }
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::unweighted_average(typename garf_types<LabelT>::matrix & predictions_out_ref,
                                                                    typename garf_types<LabelT>::matrix & variances_out_ref,
                                                                    training_set_idx_t pred_idx,
                                                                    typename garf_types<LabelT>::matrix const & outputs_per_tree_mean) const {
        // Use recurrence relations for mean and variance
        // from http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf     
        const feature_idx_t num_trees_to_use = outputs_per_tree_mean.size1();
        const feature_idx_t output_dimensions = outputs_per_tree_mean.size2();
    
        for (feature_idx_t i = 0; i < output_dimensions; i++) {
            LabelT mu_n = 0.0;
            LabelT mu_n_minus_1 = 0.0; // mean, mean at previous iteration

            LabelT sum_x_sq = 0.0; // um of all data squared - to calculate variances

            LabelT x_i; // current piece of data

            for (tree_idx_t j = 0; j < num_trees_to_use; j++) {
                // Get the current bit of data and the weight
                x_i = outputs_per_tree_mean(j, i);

                // Update mean. The +1 in the denominator is needed as otherwise
                // on the first iteration we would divide by zero
                mu_n = mu_n_minus_1 + (1.0 / (j + 1))*(x_i  - mu_n_minus_1);

                // Update running totals of sum of squares - this is for the variance
                sum_x_sq += (x_i * x_i);

                // copy values across where we have n-1 variables
                mu_n_minus_1 = mu_n;
            }

            predictions_out_ref(pred_idx, i) = mu_n;
            // Variance is sum 1/n * sum( x^2) - (mu ^ 2)
            variances_out_ref(pred_idx, i) = (sum_x_sq / num_trees_to_use) - (mu_n * mu_n);
        }
    }    

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::weighted_average(typename garf_types<LabelT>::matrix & predictions_out_ref,
                                                                    typename garf_types<LabelT>::matrix & variances_out_ref,
                                                                    training_set_idx_t pred_idx,
                                                                    typename garf_types<LabelT>::matrix const & outputs_per_tree_mean,
                                                                    typename garf_types<LabelT>::matrix const & outputs_per_tree_var) const {
        // Use recurrence relations for weight mean and variance
        // from http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
        const feature_idx_t num_trees_to_use = outputs_per_tree_mean.size1();
        const feature_idx_t output_dimensions = outputs_per_tree_mean.size2();

     
        for (feature_idx_t i = 0; i < output_dimensions; i++) {
            LabelT mu_n = 0.0;
            LabelT mu_n_minus_1 = 0.0; // these two variables used to calculate mean
            LabelT s_n = 0.0;
            LabelT s_n_minus_1 = 0.0; // these used for variance

            LabelT v_i; // current variance
            LabelT w_i, w_sum = 0; // Current weight (precision) and sum of all weights up to now
            LabelT x_i; // current piece of data

            for (tree_idx_t j = 0; j < num_trees_to_use; j++) {
                // Get the current bit of data and the weight
                x_i = outputs_per_tree_mean(j, i);
                v_i = outputs_per_tree_var(j, i);
                if (v_i == 0) {
                    // Swap in an arbitrary small number, but mustn't be too big so that we
                    // can do a weighted average with this as one of the weights.
                    // If we failed to do this then we would get a NaN which would screw everything up
                    v_i = 0.0000001;
                }
                w_i = 1.0 / v_i;

                // Update mean
                w_sum += w_i;
                mu_n = mu_n_minus_1 + (w_i / w_sum)*(x_i  - mu_n_minus_1);

                // Update variance
                s_n = s_n_minus_1 + w_i * (x_i - mu_n_minus_1) * (x_i - mu_n);

                // copy values across where we have n-1 variables
                mu_n_minus_1 = mu_n;
                s_n_minus_1 = s_n;
            }

            check_for_NaN(mu_n, "NaN resulted in mu_n after recurrence relations");
            check_for_NaN(s_n, "NaN results in s_n after recurrence relations");

            predictions_out_ref(pred_idx, i) = mu_n;
            variances_out_ref(pred_idx, i) = (s_n / w_sum);
        }
    }


#ifdef BUILD_PYTHON_BINDINGS

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::train_py(pyublas::numpy_matrix<FeatT> feature_matrix, pyublas::numpy_matrix<LabelT> label_matrix) {

        // If we have a different number of entries in features and labels, we have a problem
        if (feature_matrix.size1() != label_matrix.size1()) {
            throw std::invalid_argument("We must have the same number of features and labels");
        }

        boost::shared_ptr<typename garf_types<FeatT>::matrix> _feature_matrix(new typename garf_types<FeatT>::matrix(feature_matrix.size1(), feature_matrix.size2()));
        _feature_matrix->assign(feature_matrix);
        boost::shared_ptr<typename garf_types<LabelT>::matrix> _label_matrix(new typename garf_types<LabelT>::matrix(label_matrix.size1(), label_matrix.size2()));
        _label_matrix->assign(label_matrix);

        // allocate the training set
        boost::shared_ptr<multi_supervised_regression_training_set<FeatT, LabelT> > 
            _training_set(new multi_supervised_regression_training_set<FeatT, LabelT>(_feature_matrix, _label_matrix));

        std::cout << "regression_forest::train received " << _training_set->num_training_samples() << " samples with " 
            << _training_set->feature_dimensionality() << " feature dimensions and "
            << _training_set->inf_gain_dimensionality() << " label dimensions." << std::endl;

        train(_training_set);
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::predict_py_var_leaf_indices_num_trees_max_depth(pyublas::numpy_matrix<FeatT> feature_matrix, 
                                                                                                   pyublas::numpy_matrix<LabelT> predictions,
                                                                                                   pyublas::numpy_matrix<LabelT> variances,
                                                                                                   pyublas::numpy_matrix<node_idx_t> leaf_node_indices,
                                                                                                   tree_idx_t num_trees_to_use,
                                                                                                   depth_t max_depth,
                                                                                                   bool use_weighted_average) {
        predict(feature_matrix, &predictions, &variances, &leaf_node_indices, num_trees_to_use, max_depth, use_weighted_average);
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    void regression_forest<FeatT, LabelT, SplitT>::predict_py_var_num_trees_max_depth(pyublas::numpy_matrix<FeatT> feature_matrix, 
                                                                                  pyublas::numpy_matrix<LabelT> predictions,
                                                                                  pyublas::numpy_matrix<LabelT> variances,
                                                                                  tree_idx_t num_trees_to_use,
                                                                                  depth_t max_depth,
                                                                                  bool use_weighted_average) {
        predict(feature_matrix, &predictions, &variances, NULL, num_trees_to_use, max_depth, use_weighted_average);
    }

    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    regression_tree<FeatT, LabelT, SplitT> const& regression_forest<FeatT, LabelT, SplitT>::get_tree(tree_idx_t idx) const {
        if (idx < _stats._num_trees) {
            return _trees[idx];
        }
        throw std::invalid_argument("index of tree required is not within range");
    }
    
    
#endif
    
    
}

#endif