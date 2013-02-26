#ifndef GARF_UTIL_ARRAY_UTILS_HPP
#define GARF_UTIL_ARRAY_UTILS_HPP

namespace garf { namespace util {

	// Use the Knuth 
	template<typename T>
	void randomly_permute_column(feature_mtx<T> * const mtx, feat_idx_t f, RngSource * const rng) {
		datapoint_idx_t num_datapoints = mtx->rows();

		T temp_val;
		for (datapoint_idx_t i = 1; i < num_datapoints; i++) {
			std::uniform_int_distribution<datapoint_idx_t> swap_index_generator(0, i);

			datapoint_idx_t j = swap_index_generator(*rng);

			// Swap i and j
			temp_val = mtx->coeff(i, f);
			mtx->coeffRef(i, f) = mtx->coeffRef(j, f);
			mtx->coeffRef(j, f) = temp_val;
		}
	}

	template<typename T>
    error_t mean_squared_error(const label_mtx<T> & ground_truth_labels,
                               const label_mtx<T> & predicted_labels) {
        // Calculate the differences..
        label_mtx<T> diff = ground_truth_labels - predicted_labels;

        // sum of squared differences
        error_t sse = diff.array().square().sum();

        // divide by the number of examples we had
        return sse / ground_truth_labels.rows();
    }
}}


#endif