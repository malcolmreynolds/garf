#ifndef GARF_DECISION_FOREST
#define GARF_DECISION_FOREST

#include <boost/utility.hpp>
#include <boost/random/mersenne_twister.hpp>


#include <stdint.h>
#include <limits>

#include "forest_params.hpp"
#include "utils.hpp"

namespace garf {
    
    class decision_forest {
        clock_t _training_time;
    public:
        inline decision_forest() { };
        inline ~decision_forest() { };
        inline clock_t training_time() const { return _training_time; }
        inline void set_training_time(clock_t new_training_time) { _training_time = new_training_time; }
    };
    
    class decision_tree {
        
        tree_idx_t _tree_id;
        node_idx_t _next_free_node_id;

    public:
        
        decision_tree();
        node_idx_t get_next_node_id();
        void set_tree_id(tree_idx_t new_tree_id) { _tree_id = new_tree_id; }
        inline tree_idx_t tree_id() const { return _tree_id; };
        inline node_idx_t next_free_node_id() const { return _next_free_node_id; };

        boost::shared_ptr<boost::random::mt19937> _rng;
    };
    
    class decision_node {
        training_set_idx_t _num_samples_landing_here; // length of _samples_landing_here
        uint32_t _depth;     // depth of the node. Root node is depth zero
        node_idx_t _node_id; // Node id.. not sure exactly how necessary this is, but hey
        
        // This matrix is allocated by the parent function during training, but we
        // own it so. This stores indices to elements of the data which landed here.
        boost::shared_ptr<const tset_indices_t> _samples_landing_here;
        
        // Information gain when we split at this node. If this is -Inf then we know
        // that no split took place here (ie we are at a leaf)
        inf_gain_t _information_gain;
         
    public:
        inline decision_node(uint32_t depth) : _depth(depth), _information_gain(-std::numeric_limits<inf_gain_t>::infinity()) { };
        
        bool termination_conditions_met(const forest_params * const params) const;
        
        void set_samples_landing_here(const tset_indices_t * samples_landing_here);
        
        inline garf_types<training_set_idx_t>::vector samples_landing_here() const { return *_samples_landing_here; };
        inline training_set_idx_t num_samples_landing_here() const { return _num_samples_landing_here; };
        inline uint32_t depth() const { return _depth; };

        inline node_idx_t node_id() const { return _node_id; };
        inline void set_node_id(node_idx_t new_node_id) { _node_id = new_node_id; }

        inline inf_gain_t information_gain() const { return _information_gain; }
        inline void set_information_gain(inf_gain_t new_inf_gain) { _information_gain = new_inf_gain; }
    };
}

#include "decision/decision_tree_impl.hpp"
#include "decision/decision_node_impl.hpp"


#endif 