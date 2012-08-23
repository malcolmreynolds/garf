#ifndef GARF_TYPES_HPP
#define GARF_TYPES_HPP
    
#include <inttypes.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#ifdef BUILD_PYTHON_BINDINGS
#include <boost/python.hpp>
#include <pyublas/numpy.hpp>
#endif


namespace garf {

#ifdef BUILD_PYTHON_BINDINGS
    // see http://www.gotw.ca/gotw/079.htm for what I'm up to here
//    #warning compiling numpy pyublas arrays
    template<class ElementType>
    struct garf_types {
        typedef pyublas::numpy_matrix<ElementType> matrix;
        typedef pyublas::numpy_vector<ElementType> vector;
    };

#else
//    #warning compiling native boost arrays
    template<class ElementType>
    struct garf_types {
        typedef boost::numeric::ublas::matrix<ElementType> matrix;
        typedef boost::numeric::ublas::vector<ElementType> vector;
    };

#endif

    typedef boost::numeric::ublas::permutation_matrix<std::size_t> perm_matrix;
 
    // typedefs in case I ever need to change one of these to 64 bit
    typedef uint32_t training_set_idx_t;
    typedef uint32_t tree_idx_t;
    typedef uint32_t node_idx_t;
    typedef uint32_t feature_idx_t;
    typedef uint32_t label_idx_t;
    typedef uint32_t split_idx_t;
    typedef uint32_t depth_t;
    
    typedef double inf_gain_t;
    
    // type for a vector pointing to specific elements of the training set
    typedef garf_types<training_set_idx_t>::vector tset_indices_t;
    
    // Different types of split we can do
	enum forest_split_type_t {
	    AXIS_ALIGNED, // axis aligned - ie threshold one variable
	    LINEAR_2D, // Make a hyperplane using 2 dimensions
	    CONIC // 
    };
    
    // Encodes the splitting direction at each node of a tree (could technically expand
    // to use more than binary trees, but bun that...)
    enum split_direction_t {
        LEFT,
        RIGHT
    };
    
    // different types of affinity matrix to compute
    enum affinity_distances_types_t {
        BINARY,
        GAUSSIAN
    };
}



#endif