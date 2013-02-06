#include <stdexcept>
// #include <glog/logging.h>

namespace garf {

    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::train(const feature_matrix & features, const label_matrix & labels) {
        if (is_trained) {
            throw std::invalid_argument("forest is already trained");
        }

        uint32_t num_datapoints = features.rows();
        uint32_t data_dimensions = features.cols();
        uint32_t label_dimensions = labels.cols();

        if (labels.rows() != num_datapoints) {
            throw std::invalid_argument("number of labels doesn't match number of features");
        }

        // LOG(INFO)
        std::cout << "Forest[" << this << "] got " << num_datapoints << "x "
            << data_dimensions << " dimensional datapoints with "
            << label_dimensions << " dimensional labels" << std::endl;

        forest_stats.label_dimensions = label_dimensions;
        forest_stats.data_dimensions = data_dimensions;
        forest_stats.num_training_datapoints = num_datapoints;

        trees.reset(new RegressionTree<SplitT, SplFitterT>[forest_options.max_num_trees]);
        // LOG(INFO)
        std::cout << "created " << forest_options.max_num_trees << " trees" << std::endl;

        for (uint32_t tree_idx = 0; tree_idx < forest_options.max_num_trees; tree_idx++) {
            trees[tree_idx].tree_id = tree_idx;

            indices_vector data_indices(num_datapoints);
            if (forest_options.bagging) {
                throw std::logic_error("bagging not supported yet");
            } else {
                // gives us a vector [0, 1, 2, 3, ... num_data_points-1]
                data_indices.setLinSpaced(num_datapoints, 0, num_datapoints - 1);
            }
            trees[tree_idx].train(features, labels, data_indices, tree_options, split_options);
        }
    }



}