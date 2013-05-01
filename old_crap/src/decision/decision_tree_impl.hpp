#ifndef GARF_DECISION_TREE_IMPL_HPP
#define GARF_DECISION_TREE_IMPL_HPP

#include <boost/nondet_random.hpp>


namespace garf {

    decision_tree::decision_tree() : _next_free_node_id(0) {
#ifdef VERBOSE        
        std::cout << "decision_tree() at " << this << " - ";
#endif
        
        // Make a new mersenne twister for this tree, then seed it with proper randomness
        _rng.reset(new boost::random::mt19937);
#ifndef DEACTIVATE_RANDOMNESS
        boost::random_device rd;
        _rng->seed(rd());
#endif
#ifdef VERBOSE
        std::cout << "initialised Mersenne Twister at " << _rng.get() << std::endl;
#endif        
    }

    // Auto increments so each node being built calls this and gets a unique id for the
    // tree. Somewhat obviously, if we start building a single tree in parallel this
    // will require rethinking
    node_idx_t decision_tree::get_next_node_id() {
        _next_free_node_id++;
        return (_next_free_node_id-1);
    }
}

#endif