#ifndef GARF_REGRESSION_FOREST_HPP
#define GARF_REGRESSION_FOREST_HPP


#include <limits>
#include <iostream>

#include <boost/shared_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/utility.hpp>

#ifdef BUILD_PYTHON_BINDINGS
#include <pyublas/numpy.hpp>
#endif

#ifdef USING_OPENMP
#include <omp.h>
#endif

#ifdef USING_TBB
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/atomic.h"
using namespace tbb;
#endif

#include "decision_forest.hpp"
#include "training_set.hpp"
#include "forest_utils.hpp"
#include "forest_params.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "split_finder.hpp"



namespace garf {

    /* Forward declarations needed because of the default template argument */
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT = axis_aligned_split_finder>
    class regression_forest;
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT = axis_aligned_split_finder>
    class regression_tree;
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT = axis_aligned_split_finder>
    class regression_node;


    /* Regression forest. Given some supervised training set, we build a model which can predict outputs
       for new inputs */
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    class regression_forest : public decision_forest {
        // Training set, stored in a shared pointer so that whatever else happens, the training set
        // should not be deleted by accident while the forest still exists
        boost::shared_ptr<const supervised_training_set<FeatT, LabelT> > _training_set;

        // Parameters. These should be set before training
        regression_forest_params _params;

        // Statistics. These will be filled in as the forest trains
        regression_forest_stats _stats;

        // All the trees stored in an array
        boost::shared_array<regression_tree<FeatT, LabelT, SplitT> > _trees;

        // Different methods of averaging over all the trees - these should never be used
        // except via the predict() function
        void unweighted_average(typename garf_types<LabelT>::matrix & predictions_out_ref,
                                typename garf_types<LabelT>::matrix & variances_out_ref,
                                training_set_idx_t pred_idx,
                                typename garf_types<LabelT>::matrix const & outputs_per_tree_mean) const;
        void weighted_average(typename garf_types<LabelT>::matrix & predictions_out_ref,
                              typename garf_types<LabelT>::matrix & variances_out_ref,
                              training_set_idx_t pred_idx,
                              typename garf_types<LabelT>::matrix const & outputs_per_tree_mean,
                              typename garf_types<LabelT>::matrix const & outputs_per_tree_var) const;

        // Work out which tree indices we are going to use to predict with
        void prepare_tree_indices(typename garf_types<tree_idx_t>::vector trees_to_use,
                                  tree_idx_t num_trees_to_use,
                                  bool random_tree_permutations) const;

    public:

        // Default constructor
        regression_forest();

        // Constructor with custom parameters
        regression_forest(regression_forest_params params);
#ifdef VERBOSE
        inline ~regression_forest() { std::cout << "~regression_forest()" << std::endl; };
#endif

        // Train the forest given some training set
        void train(boost::shared_ptr<const supervised_training_set<FeatT, LabelT> > training_set);

        // Predict the answer for a set of data - return the answers and variances in
        // the provided matrices
        void predict(typename garf_types<FeatT>::matrix const & feature_matrix,
                     typename garf_types<LabelT>::matrix* predictions,
                     typename garf_types<LabelT>::matrix* variances = NULL,
                     typename garf_types<node_idx_t>::matrix* leaf_node_indices = NULL,
                     tree_idx_t num_trees_to_use = 0,
                     depth_t max_depth = std::numeric_limits<depth_t>::max(),
                     bool use_weighted_average = true);



        // Get the out-of-bag error which defines how well our regression is doing
        LabelT oob_error() const;
                     
        // Read only access to the private members
        inline const regression_forest_params* get_params() const { return &_params; };
        inline const regression_forest_stats* get_stats() const { return &_stats; };
        inline const supervised_training_set<FeatT, LabelT> & get_training_set() const { return *_training_set; };
        
#ifdef BUILD_PYTHON_BINDINGS
        // Python bindings for training
        void train_py(pyublas::numpy_matrix<FeatT> feature_matrix, pyublas::numpy_matrix<LabelT> label_matrix);
        
        // Using various separate functions here, but all bound to "predict" on python side
        // void predict_py_no_var(pyublas::numpy_matrix<FeatT> feature_matrix, 
        //                        pyublas::numpy_matrix<LabelT> predictions);
        // void predict_py_var_leaf_indices(pyublas::numpy_matrix<FeatT> feature_matrix,
        //                                  pyublas::numpy_matrix<LabelT> predictions,
        //                                  pyublas::numpy_matrix<LabelT> variances,
        //                                  pyublas::numpy_matrix<node_idx_t> leaf_node_idx);
        void predict_py_var_num_trees_max_depth(pyublas::numpy_matrix<FeatT> feature_matrix,
                                                pyublas::numpy_matrix<LabelT> predictions,
                                                pyublas::numpy_matrix<LabelT> variances,
                                                tree_idx_t num_trees_to_use,
                                                depth_t max_depth,
                                                bool use_weighted_average);                                
        void predict_py_var_leaf_indices_num_trees_max_depth(pyublas::numpy_matrix<FeatT> feature_matrix,
                                                             pyublas::numpy_matrix<LabelT> predictions,
                                                             pyublas::numpy_matrix<LabelT> variances,
                                                             pyublas::numpy_matrix<node_idx_t> leaf_node_idx,
                                                             tree_idx_t num_trees_to_use,
                                                             depth_t max_depth,
                                                             bool use_weighted_average);

        // Simple getters and setters
        inline regression_forest_params const& get_params_py() const { return _params; };
        void set_params_py(regression_forest_params params) { _params = params; }
        inline regression_forest_stats const& get_stats_py() const { return _stats; };
        inline regression_tree<FeatT, LabelT, SplitT> const& get_tree(tree_idx_t idx) const;
#endif
    };
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    class regression_tree : public decision_tree {
        // root node
        boost::shared_ptr<regression_node<FeatT, LabelT, SplitT> > _root;

        //we don't own the forest / params, so no boost pointer here.
        const regression_forest<FeatT, LabelT, SplitT>* _forest;
        const regression_forest_params * _params;
        const regression_forest_stats * _stats;
    public:
        inline regression_tree() : decision_tree() { }
#ifdef VERBOSE       
        inline ~regression_tree() { std::cout << "~regression_tree() id " << this->tree_id() << std::endl; }
#endif
        void init(const regression_forest<FeatT, LabelT, SplitT>* forest, tree_idx_t tree_id);
        void train(const supervised_training_set<FeatT, LabelT> & training_set);
        // starting to use the convention here that out parameters are passed in with pointers, so that
        // at the call site we see a &
        node_idx_t predict(const matrix_row<const typename garf_types<FeatT>::matrix> & sample, 
                           typename garf_types<LabelT>::matrix::iterator2* prediction_mean,
                           typename garf_types<LabelT>::matrix::iterator2* prediction_var) const;
        node_idx_t predict(const matrix_row<const typename garf_types<FeatT>::matrix> & sample, 
                           typename garf_types<LabelT>::matrix::iterator2* prediction_mean,
                           typename garf_types<LabelT>::matrix::iterator2* prediction_var,
                           depth_t max_depth) const;

        // Get max depth over all children
        node_idx_t max_depth() const;

        regression_node<FeatT, LabelT, SplitT> const& get_root() const;

#ifdef BUILD_PYTHON_BINDINGS
        inline regression_forest<FeatT, LabelT, SplitT> const& get_forest() const { return *_forest; }
#endif
    };
    
    template<typename FeatT, typename LabelT, template <typename, typename> class SplitT>
    class regression_node : public decision_node {
        // Store left and right subtrees
        boost::shared_ptr<regression_node<FeatT, LabelT, SplitT> > _left;
        boost::shared_ptr<regression_node<FeatT, LabelT, SplitT> > _right;
        
        // Splits the datapoints landing at this node. We have a template
        // parameter on this so we can switch between axis aligned, pairwise etc without
        // speed penalty
        boost::shared_ptr< SplitT<FeatT, LabelT> > _splitter;
    
        boost::shared_ptr<const multivariate_normal<LabelT> > _sample_distribution; // multivariate gaussian representing the label distribution of data at this node
        
        // These shouldn't be changed once the tree_node is initialised
        const regression_node<FeatT, LabelT, SplitT>* const _parent;
        // tree cannot be a const pointer because we need to call get_next_node_id which cannot be a const function
        regression_tree<FeatT, LabelT, SplitT>* const _tree;

        // Returns true if all the variances within the sample distribution
        // are lower than the provided minimum
        bool variance_below_minimum(const regression_forest_params* const params) const;
        
    public:    
        regression_node(const regression_node<FeatT, LabelT, SplitT>* const parent,
                        regression_tree<FeatT, LabelT, SplitT>* const tree);
        ~regression_node();
        
        void train(const supervised_training_set<FeatT, LabelT> & training_set,
                   const tset_indices_t * samples_landing_here,
                   const multivariate_normal<LabelT> * sample_distribution,
                   boost::random::mt19937& rng,
                   const regression_forest_params * const params);
                   
        node_idx_t max_depth() const;

        inline multivariate_normal<LabelT> const& get_sample_distribution() const { return *_sample_distribution; };
        inline SplitT<FeatT, LabelT> const& get_splitter() const; // { return *_splitter; }
        inline bool is_root() const { return _parent == NULL; }
        inline bool is_internal() const { return !is_leaf(); }
        inline bool is_leaf() const {  return (_left.get() == NULL); } // left being null implies right is null as well, unless we have serious bugs
        inline regression_node<FeatT, LabelT, SplitT> const& get_left() const;
        inline regression_node<FeatT, LabelT, SplitT> const& get_right() const;
        inline boost::shared_ptr<regression_node<FeatT, LabelT, SplitT> > left() const { return _left; };
        inline boost::shared_ptr<regression_node<FeatT, LabelT, SplitT> > right() const { return _right; };    
    };
    

    
}

// Implementation details in separate header files
#include "regression/regression_forest_impl.hpp"
#include "regression/regression_tree_impl.hpp"
#include "regression/regression_node_impl.hpp"


#endif