#ifndef GARF_DECISION_NODE_IMPL_HPP
#define GARF_DECISION_NODE_IMPL_HPP

namespace garf {
    bool decision_node::termination_conditions_met(const forest_params* const params) const {
        // Termination condition 1 - check if we have too few data points
        if (num_samples_landing_here() <= params->_min_sample_count) {
#ifdef VERBOSE
            std::cout << num_samples_landing_here() << " samples landed at node, stopping growing" << std::endl;
#endif
            return true;
        }
        // Termination condition 2 - check if we have reached the maximum depth
        if (_depth >= params->_max_tree_depth) {
            //if we have gone over the correct depth, something is broken in my logic
            if (_depth > params->_max_tree_depth) {
                throw new std::logic_error("max_tree_depth overrun! This code sucks!");
            }
#ifdef VERBOSE
            std::cout << "maximum depth of " << params->_max_tree_depth << " reached, stopping growing" << std::endl;
#endif
            return true;
        }

        // Check if every piece of data at this node is identical - this may 
        // be the case when bagging is enabled, so even though the number of
        // datapoints landing here is > params->_min_sample_count, this data
        // is actually identical and hence splitting it is pointless
        if (all_elements_identical<training_set_idx_t>(*_samples_landing_here)) {
            return true;
        }

        return false;
    }

        
    void decision_node::set_samples_landing_here(const tset_indices_t * samples_landing_here) {
        _samples_landing_here.reset(samples_landing_here);
        _num_samples_landing_here = _samples_landing_here->size();
    }

}

#endif