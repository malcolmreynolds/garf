#ifndef GARF_SMART_HYPERPLANE_SPLIT_FINDER_IMPL_HPP
#define GARF_SMART_HYPERPLANE_SPLIT_FINDER_IMPL_HPP

#include <cmath>

namespace garf {

	template<typename FeatT, typename LabelT>
	smart_hyperplane_split_finder<FeatT, LabelT>::smart_hyperplane_split_finder(const training_set<FeatT> & training_set,
											                                    const tset_indices_t& valid_indices,
											                                    const multivariate_normal<LabelT> & parent_dist,
											                                    training_set_idx_t num_in_parent,
    										                                    uint32_t num_splits_to_try, uint32_t num_threshes_per_split)
	  : hyperplane_split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, 
			                                   num_in_parent, num_splits_to_try, num_threshes_per_split) {
	}

    // Generate the hyperplane directions. Note that we don't use the feature_indices in this instance
    template<typename FeatT, typename LabelT>
    void smart_hyperplane_split_finder<FeatT, LabelT>::generate_hyperplane_coefficients(typename garf_types<FeatT>::matrix & plane_coeffs,
                                                                                  const typename garf_types<feature_idx_t>::matrix & feature_indices,
                                                                                  boost::random::mt19937& gen) {

    	// std::cout << "inside smart_hyperplane_split_finder::generate_hyperplane_coefficients on " << this->name() << std::endl;

        // for each hyperplane we are going to generate, we need to pick 2 datapoints and generate 
        // the normal vector (in this subset of dimensions) between them. then we subsample only the relevant
        // feature indices from it

    	// Need a uniform distribution to pick elements of the training set
        uniform_int_distribution<> idx_dist(0, (this->_num_training_samples - 1));

        for (uint32_t i=0; i < this->num_splits_to_try(); i++) {
        	training_set_idx_t data_idx_a = this->_valid_indices(idx_dist(gen));
        	training_set_idx_t data_idx_b = data_idx_a;
        	while (data_idx_a == data_idx_b) {  // Cannot allow both datapoints to be same, then there is no index between them
        		data_idx_b = this->_valid_indices(idx_dist(gen));
        	}

        	// We now have selected 2 different indices pointing to datapoints landing at this node.
        	// We need to get the difference vector (doesn't matter which way round we do this)
            const matrix_row<const typename garf_types<FeatT>::matrix> datapoint_a(this->_training_set.features(), data_idx_a);
            const matrix_row<const typename garf_types<FeatT>::matrix> datapoint_b(this->_training_set.features(), data_idx_b);

        	typename garf_types<FeatT>::vector difference_vector = datapoint_a - datapoint_b;

        	// Copy the relevant dimensions only into the plane coeffs. sum_so_far stores
        	// a running total of the magnitude of the vector, so we can normalise it
        	double sum_so_far = 0.0;
        	for (uint32_t j=0; j < this->get_hyperplane_dimensionality(); j++) {
        		FeatT v = difference_vector(feature_indices(i, j));
        		plane_coeffs(i, j) = v;
        		sum_so_far += (v * v);
        	}

        	// Normalise
        	double hyperplane_vector_mag = sqrt(sum_so_far);
        	for (uint32_t j=0; j < this->get_hyperplane_dimensionality(); j++) {
        		plane_coeffs(i, j) /= hyperplane_vector_mag;
        	}
        }
    }
}

#endif