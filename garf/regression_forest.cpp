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

        std::cout << "Forest[" << this << "] got " << num_datapoints << "x "
            << data_dimensions << " dimensional datapoints with "
            << label_dimensions << " dimensional labels" << std::endl;

        forest_stats.label_dimensions = label_dimensions;
        forest_stats.data_dimensions = data_dimensions;
        forest_stats.num_training_datapoints = num_datapoints;

        trees.reset(new RegressionTree<SplitT, SplFitterT>[forest_options.max_num_trees]);
        forest_stats.num_trees = forest_options.max_num_trees;
        std::cout << "created " << forest_stats.num_trees << " trees" << std::endl;

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

        // We are done, so set the forest as trained
        is_trained = true;
    }

    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::clear() {
        std::cout << "clearing forest of " << forest_stats.num_trees << " trees." << std::endl;
        trees.reset();
        forest_stats.num_trees = 0;
        is_trained = false;
    }


    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::predict(const feature_matrix & features,
                                                       label_matrix * const output_labels,
                                                       label_matrix * const output_variance) const {
        feat_idx_t num_features_to_predict = features.rows();
        if (!is_trained) {
            throw std::invalid_argument("cannot predict, forest not trained yet");
        } else if (features.cols() != forest_stats.data_dimensions) {
            throw std::invalid_argument("predict(): feature_matrix.cols != trained data dimensions");
        } else if (output_labels->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): output_labels->cols() != trained label dimensions");
        } else if (output_variance->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): output_variance->cols() != trained label dimensions");
        } else if (output_labels->rows() != num_features_to_predict) {
            throw std::invalid_argument("predict(): output_label->rows() doesn't match num features to predict");
        } else if (output_variance->rows() != num_features_to_predict) {
            throw std::invalid_argument("predict(): output_variance->rows() doesn't match num features to predict");
        }

        std::cout << "in predict(), tests passed" << std::endl;


        for (feat_idx_t feat_vec_idx = 0; feat_vec_idx < num_features_to_predict; feat_vec_idx++) {
            // for each datapoint, we want to work out the set of leaf nodes it reaches
        }


    }

}