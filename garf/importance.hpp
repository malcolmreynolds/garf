#ifndef GARF_IMPORTANCE_HPP
#define GARF_IMPORTANCE_HPP

// Functions to do the proper feature importance testing, as defined in the Breiman paper

namespace garf {

    // Calculate vari
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForesT<FeatT, LabT, SplitT, SplFitterT>::calculate_feature_importance(const feature_mtx<FeatT> & features,
                                                                                         const label_mtx<L> & labels,
                                                                                         importance_vec & importance_out) {



        // 1. Need to make a local (stack allocated if possible) copy the size of the feature matrix in
        // For each tree:
        //     Make a mask of datapoins which are out of bag for the tree
        //     Predict with tree for each of these datapoints
        //     for each variable:
        //         permute the values in these variables, put through the forest

        feat_idx_t num_features = forest_stats.data_dimensions;
        if (importance_out.size() != num_features) {
            throw std::invalid_argument("importance_out vector is wrong shape.")
        } else if (features.rows() != forest_stats.num_training_datapoints) {
            throw std::invalid_argument("features matrix has wrong # of rows: must be the exact features we trained with!");
        } else if (features.cols() != forest_stats.data_dimensions) {
            throw std::invalid_argument("features matrix has wrong # of cols: must be the exact features we trained with!");
        } else if (labels.rows() != forest_stats.num_training_datapoints) {
            throw std::invalid_argument("labels matrix has wrong # of rows: must be the exact labels we trained with!");
        } else if (labels.cols() != forest_stats.label_dimension) {
            throw std::invalid_argument("labels matrix has wrong # of cols: must be the exact labels we trained with!");
        }

        bool_vec samples_are_out_of_bag(forest_stats.num_training_datapoints);

        tree_idx_t num_trees = forest_stats.num_trees;
        for (tree_idx_t t = 0; t < num_trees; t++) {
            // Work out which datapoints are out of bag - initialise all to true
            samples_are_out_of_bag.setOnes();

            // Make out of bag mask
            datapoint_idx_t samples_in_tree = trees[t].root->num_samples();
            std::cout << "[t" << t.tree_id << "] received " << samples_in_tree << " (not necessarily distinct) datapoints" << std::endl;
            const data_indices_vec & root_node_samples = trees[t].root.training_data_indices;
            for (datapoint_idx_t d = 0; d < forest_stats.num_training_datapoints; d++) {
                // Each datapoint that passed through the root node of a tree is by definition
                // not out of bag
                samples_are_out_of_bag(root_node_samples(d)) = false;
            }

            // Count the number that are out of bag
            datapoint_idx_t num_out_of_bag = samples_are_out_of_bag.sum();
            std::cout << "[t" << t.tree_id << "] has " << num_out_of_bag << " out of bag" << std::endl;

            // Copy out of bag features and labels to separate array
            feature_mtx<FeatT> oob_features(num_out_of_bag, num_features);
            label_mtx<LabT> oob_labels(num_out_of_bag, forest_stats.label_dimension);

            // Scan through the samples_are_out_of_bag array, for everything that *is* out
            // of bag, copy it to the oob_features / oob_labels array
            datapoint_idx_t next_free_row = 0;
            for (datapoint_idx_t d = 0; d < forest_stats.num_training_datapoints; d++) {
                if (samples_are_out_of_bag(d)) {
                    oob_features.row(next_free_row) = features.row(d);
                    oob_labels.row(next_free_row) = labels.row(d);
                    next_free_row++;
                }
            }

            error_t oob_error = test_error(oob_features, oob_labels);

            // Now need a feature vector which basically stores a column of the feature matrix.
            feature_vec<FeatT> saved_column(num_out_of_bag);
            feature_vec<error_t> error_with_each_feature_permuted(num_features);

            for (feat_idx_t f = 0; f < num_features; f++) {

                // Save the contents of feature f, permute them randomly, then get the modified error
                saved_column = oob_features.col(f);

                randomly_permute_column(oob_features, f);

                error_with_each_feature_permuted(f) = test_error(oob_features, oob_labels);

                // Restore the jumbled features
                oob_features.col(f) = saved_column

            }
        }
    }

    template<typename T>
    static void randomly_permute_column(feature_mtx<T> & features, )




}



#endif