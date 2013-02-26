#ifndef GARF_IMPORTANCE_HPP
#define GARF_IMPORTANCE_HPP

#include "util/array_utils.hpp"

// Functions to do the proper feature importance testing, as defined in the Breiman paper

namespace garf {

    // Calculate vari
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::calculate_feature_importance(const feature_mtx<FeatT> & features,
                                                                                         const label_mtx<LabT> & labels,
                                                                                         importance_vec * const importance_out) {



        // 1. Need to make a local (stack allocated if possible) copy the size of the feature matrix in
        // For each tree:
        //     Make a mask of datapoins which are out of bag for the tree
        //     Predict with tree for each of these datapoints
        //     for each variable:
        //         permute the values in these variables, put through the forest

        feat_idx_t num_features = forest_stats.data_dimensions;
        if (importance_out->size() != num_features) {
            throw std::invalid_argument("importance_out vector is wrong shape.");
        } else if (!feature_mtx_correct_shape(features, forest_stats.num_training_datapoints)) {
            throw std::invalid_argument("features matrix has wrong shape: must be the exact features we trained with!");
        } else if (!label_mtx_correct_shape(labels, forest_stats.num_training_datapoints)) {
            throw std::invalid_argument("labels matrix has wrong shape: must be the exact labels we trained with!");
        }

        RngSource rng;
        bool_vec samples_are_out_of_bag(forest_stats.num_training_datapoints);
        tree_idx_t num_trees = forest_stats.num_trees;
        datapoint_idx_t num_out_of_bag;

        // Want to make temporary arrays instead of creating oob_{features,labels}
        // inside loop. we don't know how tall these arrays need to be, so we make
        // them the height of num_training_datapoints - 1. This is because (technically)
        // all but one of the datapoints could have been out of bag for a particular run.
        // NB: this is fucking unlikely, but hey.
        feature_mtx<FeatT> oob_features(forest_stats.num_training_datapoints-1, num_features);
        label_mtx<LabT> oob_labels(forest_stats.num_training_datapoints-1,
                                   forest_stats.label_dimensions);

        // This is a working matrix which we pass into tree::test_error. Predictions are put into here so
        // that the MSE can be calculated. Technically we could skip allocating this here
        // but then it would be allocated loads of times in that function
        label_mtx<LabT> predicted_labels_tmp(forest_stats.num_training_datapoints-1,
                                             forest_stats.label_dimensions);

        // Need to store all the results somewhere
        error_mtx error_increase_with_tree_feature(num_trees, num_features);

        for (tree_idx_t t = 0; t < num_trees; t++) {
            // std::cout << "doing importance for tree " << t << std::endl;

            // Work out which datapoints are out of bag - initialise all to true
            samples_are_out_of_bag.setOnes();

            // Make out of bag mask
            const datapoint_idx_t samples_in_tree = trees[t].get_root().num_samples();
            const data_indices_vec & root_node_samples = trees[t].get_root().training_data_indices;
            num_out_of_bag = samples_in_tree;
                
            // std::cout << "[t" << t << "]: in bag: " << root_node_samples.transpose() << std::endl;

            for (datapoint_idx_t d = 0; d < forest_stats.num_training_datapoints; d++) {
                if (samples_are_out_of_bag(root_node_samples(d))) {
                    // If we are here then the current datapoint was previously thought
                    // to be out of bag. It is actually not. Therefore we decrement the number
                    // of out of bag datapoints we have (which started off at the full number),
                    // and mark it as being not out of bag so we don't count it again.
                    num_out_of_bag--;
                    samples_are_out_of_bag(root_node_samples(d)) = false;
                }
            }

            // Scan through the samples_are_out_of_bag array, for everything that *is* out
            // of bag, copy it to the top of the oob_features / oob_labels array
            datapoint_idx_t next_free_row = 0;

            // std::cout << "[t" << t << "]: out of bag: ";
            for (datapoint_idx_t d = 0; d < forest_stats.num_training_datapoints; d++) {
                if (samples_are_out_of_bag(d)) {
                    // std::cout << d << " ";
                    oob_features.row(next_free_row) = features.row(d);
                    oob_labels.row(next_free_row) = labels.row(d);
                    next_free_row++;
                }
            }
            // std::cout << std::endl;

            // Predict with all the out of bag data, just for this tree
            error_t oob_error_for_unpermuted_data = trees[t].test_error(oob_features, oob_labels, &predicted_labels_tmp,
                                                                        predict_options, num_out_of_bag);

            // Now need a feature vector which basically stores a column of the feature matrix.
            feature_vec<FeatT> saved_column(num_out_of_bag);
            feature_vec<error_t> error_with_each_feature_permuted(num_features);

            // std::cout << "[t" << t << "] samples out of bag: " << std::endl
            //     << 

            // For each feature in turn
            for (feat_idx_t f = 0; f < num_features; f++) {

                // Save the contents of feature f, permute them randomly, then get the modified error
                saved_column = oob_features.block(0, f, num_out_of_bag, 1);

                util::randomly_permute_column<FeatT>(&oob_features, f, &rng);

                error_increase_with_tree_feature(t, f) = trees[t].test_error(oob_features, oob_labels, &predicted_labels_tmp,
                                                                             predict_options, num_out_of_bag);

                // std::cout << "[t" << trees[t].tree_id << "] f#" << f << " permuted: error = " << error_with_each_feature_permuted(f) << std::endl;

                // Restore the jumbled features
                oob_features.block(0, f, num_out_of_bag, 1) = saved_column;
            }

            // After we've done all the error increase jazz, we need to
            // divide this row (ie all the different oob error rates
            // for a single tree) by the oob error we got when we hadn't shuffled the data
            error_increase_with_tree_feature.row(t) /= oob_error_for_unpermuted_data;
        }

        // Now need to average over all trees to get the final feature importance
        *importance_out = error_increase_with_tree_feature.colwise().sum();
        *importance_out /= importance_out->sum(); // normalise
    }
}



#endif