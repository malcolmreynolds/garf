#ifndef GARF_UTILS_HPP
#define GARF_UTILS_HPP

#include <iostream>
#include <sstream>

#include <boost/utility.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "types.hpp"


namespace garf {
	 
	class threadsafe_ostream {
	    std::ostringstream _std_stream;
	public:
	    threadsafe_ostream(){}
	    template <class T>
	    threadsafe_ostream& operator<<(const T& inData){
	        _std_stream << inData;
	        return *this;
	    }
	    std::string toString() const {
	        return _std_stream.str();
	    }
	};


	template<typename FeatT>
	inline bool all_elements_identical(const typename garf_types<FeatT>::vector & v) {
		
		typename garf_types<FeatT>::vector::size_type num_elements = v.size();
		if (num_elements == 0) {
			throw std::logic_error("called all_elements_identical on vector of size zero");
		}

		FeatT first_element = v(0);
		for (typename garf_types<FeatT>::vector::size_type i = 1; i < num_elements; i++) {
			if (v(i) != first_element) {
				// We have found something that is different from the first element,
				// therefore the vector contains at least two values.
				return false;
			}
		}
		return true;
	}

	template<typename T>
	inline void knuth_shuffle(typename garf_types<T>::vector & v, uint32_t num_to_shuffle, boost::mt19937 *gen) {
		T tmp;
	    uint32_t dest_idx;
	    uint32_t v_size = v.size();
        for (uint32_t i = 0; i < num_to_shuffle; i++) {
            // This could be done more efficiently than making a new RNG every time.. but this way is cleaner.
            // No point changing it until it's a bottleneck.

            // This is the Fisher-Yates / Knuth shuffle
            boost::random::uniform_int_distribution<> dist(i, v_size-1);
            dest_idx = dist(*gen);
            tmp = v(dest_idx);
            v(dest_idx) = v(i);
            v(i) = tmp;
        }
	}

    // This only makes sense for numeric types
    template<typename T>
    inline void check_for_NaN(T t, const char * message) {
        if (t != t) {
            throw std::runtime_error(message);
        }
    }

}

#endif