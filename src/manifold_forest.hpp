#ifndef GARF_MANIFOLD_FOREST_HPP
#define GARF_MANIFOLD_FOREST_HPP

#include <boost/shared_array.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/utility.hpp>

#ifdef BUILD_PYTHON_BINDINGS
//#define PYUBLAS_HAVE_BOOST_BINDINGS
#include <pyublas/numpy.hpp>
#endif

#include "training_set.hpp"

#include "decision_forest.hpp"
#include "forest_params.hpp"
#include "split_finder.hpp"
#include "forest_utils.hpp"
#include "multivariate_normal.hpp"

//need to traverse tree non-recursively
#include <stack>

#include <iostream>
#include <algorithm>

namespace garf  {
	
    //using namespace boost::numeric::ublas;
	
    // need forward declarations of stuff as it is reference by the forest
    template<typename FeatT> class manifold_tree;
    template<typename FeatT> class manifold_node;
	
    /* Manifold forests - Split a bunch of data down similar to a density forest, but
       instead of learning a probability distribution over all the data, we want to compute
       a low dimensional embedding. This necessitates computing an affinity matrix between
       the datapoints in the training set, with affinity being based on which datapoints land
       in the same leaf nodes. */
    template<typename FeatT>
    class manifold_forest : public decision_forest {
    public:
        manifold_forest();
        manifold_forest(manifold_forest_params params);
        inline ~manifold_forest() { std::cout << "~manifold_forest()" << std::endl; };
        void train(boost::shared_ptr<training_set<FeatT> > training_set);
        void compute_affinity_matrix();
#ifdef BUILD_PYTHON_BINDINGS
        void train_py(pyublas::numpy_matrix<FeatT> feature_matrix);
        void train_sup_py(pyublas::numpy_matrix<FeatT> feature_matrix, pyublas::numpy_matrix<double> label_matrix);
        pyublas::numpy_matrix<double> get_affinity_matrix_py();
        inline manifold_forest_params const& get_params_py() const { return _params; };
        void set_params_py(manifold_forest_params params) { _params = params; }

        inline manifold_forest_stats const& get_stats_py() const { return _stats; };
        inline manifold_tree<FeatT> const& get_tree(tree_idx_t idx) const;
        // inline
#endif
        inline const manifold_forest_stats* get_stats() const { return &_stats; } ;
        inline const manifold_forest_params* get_params() const { return &_params; } ;
        inline training_set<FeatT> const& get_training_set() const { return *_training_set; };

        void hello();
        
        boost::shared_ptr<garf_types<double>::matrix > _affinity_matrix;
		
    private:
        
        // Stores the training set. This can be passed in from any bit of C++, or it can be built by train_py.
        boost::shared_ptr<training_set<FeatT> > _training_set;
        
        // params is stuff that is set in advance of providing the forest data (max depth etc)
        manifold_forest_params _params;
                
        // stats is stuff that arises once data has been provided to the forest - actual (instead
        // of maximum) number of trees, etc
        manifold_forest_stats _stats;
		
        // all the trees. This is a shared_array so that the default copy constructor works - note that
        // this will NOT do a deep copy as we don't want that happening unless explicitly asked for
        boost::shared_array<manifold_tree<FeatT> > _trees;
    }; 
	
	
	
	
    template<typename FeatT>
    class manifold_tree : public decision_tree {

        // smart pointers 
        boost::shared_ptr<manifold_node<FeatT> > _root;
        boost::shared_ptr<typename garf_types<double>::matrix > _affinity_matrix;

        //we don't own the forest / params, so no boost pointer here.
        const manifold_forest<FeatT>* _forest;
        const manifold_forest_params * _params;
        const manifold_forest_stats * _stats;


    public:
        inline manifold_tree() : decision_tree() { }
        inline ~manifold_tree() { std::cout << "~manifold_tree() id "<< this->tree_id() << std::endl; };
        void init(const manifold_forest<FeatT>* forest, tree_idx_t tree_id);
        void train(training_set<FeatT>& training_set);
#ifdef BUILD_PYTHON_BINDINGS
        inline manifold_forest<FeatT> const& get_forest() const { return *_forest; }
        manifold_node<FeatT> const& get_root() const;
#endif
        const typename garf_types<double>::matrix& compute_tree_affinity_matrix();
        void compute_leaf_affinity(const manifold_node<FeatT>& node);
    };




    // should make this inherit from decision tree node but can't be bothered
    template<typename FeatT>
    class manifold_node : public decision_node {
        // We own the children (we want them to be deleted when we are deleted)
        boost::shared_ptr<manifold_node<FeatT> > _left;
        boost::shared_ptr<manifold_node<FeatT> > _right;

 

        boost::shared_ptr<split_finder<FeatT> > _splitter;

        boost::shared_ptr<multivariate_normal<FeatT> > _sample_distribution; // multivariate gaussian representing the distribution of data at this node

		// These shouldn't be changed once the tree_node is initialised
        const manifold_node<FeatT>* const _parent;
        // tree cannot be a const pointer because we need to call get_next_node_id which cannot be a const function
        manifold_tree<FeatT>* const _tree;

    public:
        manifold_node(const manifold_node<FeatT>* const parent_node,
                           manifold_tree<FeatT>* const tree);
        ~manifold_node(); 
        // Note that the training set can't be a const reference because we need to
        // call set_parent_indices. sucks, but more efficient.
        void train(training_set<FeatT>& training_set,
                   garf_types<training_set_idx_t>::vector* samples_landing_here,
                   multivariate_normal<FeatT>* sample_distribution,
                   boost::random::mt19937& rng,
                   const manifold_forest_params* const params);



        
        inline multivariate_normal<FeatT> const& get_sample_distribution() const { return *_sample_distribution; };
        inline split_finder<FeatT> const& get_splitter() const { return *_splitter; }
        inline bool is_root() const { return _parent == NULL; }
        inline bool is_internal() const { return !is_leaf(); }
        inline bool is_leaf() const { return (_left.get() == NULL); }
        inline manifold_node<FeatT> const& get_left() const {
            if (_left.get() == NULL) {
                throw std::logic_error("Node doesn't have a left child");
            }
            return *_left;
        }
        inline manifold_node<FeatT> const& get_right() const {
            if (_right.get() == NULL) {
                throw std::logic_error("Node doesn't have a right child");
            }
            return *_right;
        }
        inline boost::shared_ptr<manifold_node<FeatT> > left() const { return _left; };
        inline boost::shared_ptr<manifold_node<FeatT> > right() const { return _right; };

    } ;
    
    // Implementation of all these class methods in separate files, to keep this header file vaguely grok-able
    #include "manifold/manifold_forest_impl.hpp"
    #include "manifold/manifold_tree_impl.hpp"
    #include "manifold/manifold_node_impl.hpp"
    
}


#endif
