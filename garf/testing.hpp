#ifndef GARF_TESTING_HPP
#define GARF_TESTING_HPP

namespace garf {

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    error_t RegressionForest<FeatT, LabT, SplitT, SplFitterT>::test_error(const feature_mtx<FeatT> & features,
                                                                          const label_mtx<LabT> & ground_truth_labels) const {

        datapoint_idx_t num_datapoints = features.rows();

        // Check matrices are right size
        if (!feature_mtx_correct_shape(features, num_datapoints)) {

        } else if (!label_mtx_correct_shape(ground_truth_labels, num_datapoints)) {
            throw std::invalid_argument("ground_truth_labels is wrong shape");
        }

        // Create output array for labels
        label_mtx<LabT> predicted_labels(num_datapoints, forest_stats.label_dimensions);

        // Do the prediction
        predict(features, &predicted_labels);

        return mean_squared_error(ground_truth_labels, predicted_labels);
    }
}


#endif