#ifndef GARF_SPLITTER_HPP
#define GARF_SPLITTER_HPP

#include "types.hpp"

namespace garf {
	// Simply the parent class for all other splitters
	class Splitter {
		// Don't really need anything in here. We used to have a virtual function
		// for evaluate() but I actually want it inlined everywhere, and also we are
		// always going to be working with 
	};

	class AxisAlignedSplt : public Splitter {
	public:
		feat_idx_t feat_idx;
		double split_value;
		inline split_dir_t evaluate(const feature_vector & fvec) const {
	        if (fvec(feat_idx) >= split_value) {
	            return LEFT;
	        }
	        return RIGHT;
		}

	};
}

// #include "split/axis_aligned.hpp"

#endif