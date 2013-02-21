
// #include <glog/logging.h>

#include "util/random_seed.hpp"
#include <random>

namespace garf {

    const uint32_t random_seed_sequence_length = 32;

#ifdef GARF_PARALLELIZE_TBB
    // Used to synchronize access to cout to avoid absolute garbage (interleaved strings) on cout
    tbb::mutex cout_mutex;

// // using a temporary instead of returning one from a function avoids any issues with copies
// FullExpressionAccumulator(std::cout) << val1 << val2 << val3;

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    class concurrent_tree_trainer {
        boost::shared_array<RegressionTree<FeatT, LabT, SplitT, SplFitterT> > & trees;
        const feature_mtx<FeatT> & all_features;
        const label_mtx<LabT> & all_labels;
        const RegressionForest<FeatT, LabT, SplitT, SplFitterT> & forest;
    public:

        // Need to make this get called with a larger rane
        void operator() (const blocked_range<tree_idx_t> & r) const {

            const datapoint_idx_t num_training_datapoints = all_features.rows();
            const feat_idx_t data_dimensions = all_features.cols();
            const label_idx_t label_dimensions = all_labels.cols();

            // distribution to pick the bagging indices
            std::uniform_int_distribution<datapoint_idx_t> bagging_index_picker(0, num_training_datapoints - 1);


            // If split_options.properly_random is selected then we will build a proper random sequence in here.
            // If the user has not selected proper randomness this will be a null pointer, which we can safely 
            // pass into the fitter constructor as it won't be dereferenced
            boost::shared_ptr<std::seed_seq> seed;

            if (forest.split_options.properly_random) {
                // Make a proper random seed, reading from /dev/random
                boost::shared_array<util::seed_t> seed_sequence(util::random_seed_sequence(random_seed_sequence_length));
                seed.reset(new std::seed_seq(seed_sequence.get(), seed_sequence.get() + random_seed_sequence_length));
            }

            // // Uncomment this block to see what the random seeds are saying - this is basically
            // // just to check that the random seed sequence is giving some different values to each RNG
            // std::cout << "random seeds:" << std::endl;
            // std::vector<uint32_t> seeds(10);
            // seed->generate(seeds.begin(), seeds.end());
            // for(std::uint32_t n : seeds) {
            //     std::cout << n << std::endl;
            // }

            SplFitterT<FeatT, LabT> fitter(forest.split_options, num_training_datapoints,
                                           data_dimensions, label_dimensions, cout_mutex, seed.get());

            cout_mutex.lock();
            std::cout << this << " got range [" << r.begin() << "," << r.end() << ") grain =  " << r.grainsize() << std::endl;
            cout_mutex.unlock();

            for (tree_idx_t t = r.begin(); t != r.end(); t++) {
                data_indices_vec data_indices(num_training_datapoints);
                if (forest.forest_options.bagging) {
                    // bagging: Sample indices from full dataset WITH REPLACEMENT!!!
                    for (datapoint_idx_t d = 0; d < num_training_datapoints; d++) {
                        data_indices(d) = bagging_index_picker(fitter.rng);
                    }
                } else {
                    // vector [0, 1, 2, 3, ... num_data_points-1] => no bagging
                    data_indices.setLinSpaced(num_training_datapoints, 0, num_training_datapoints - 1);
                }

                trees[t].tree_id = t;
                trees[t].train(all_features, all_labels, data_indices, forest.tree_options, &fitter);
            }
        }

        concurrent_tree_trainer(boost::shared_array<RegressionTree<FeatT, LabT, SplitT, SplFitterT> > & _trees,
                                const feature_mtx<FeatT> & _all_features,
                                const label_mtx<LabT> & _all_labels,
                                const RegressionForest<FeatT, LabT, SplitT, SplFitterT> & _forest)
            : trees(_trees), all_features(_all_features), all_labels(_all_labels), forest(_forest) {
        }

    };

#endif


    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::train(const feature_mtx<FeatT> & features, const label_mtx<LabT> & labels) {
        if (trained) {
            throw std::invalid_argument("forest is already trained");
        }

        datapoint_idx_t num_datapoints = features.rows();
        feat_idx_t data_dimensions = features.cols();
        label_idx_t label_dimensions = labels.cols();

        if (labels.rows() != num_datapoints) {
            throw std::invalid_argument("number of labels doesn't match number of features");
        }

        std::cout << "Forest[" << this << "] got " << num_datapoints << "x "
            << data_dimensions << " dimensional datapoints with "
            << label_dimensions << " dimensional labels" << std::endl;
        std::cout << "creating " << forest_options.max_num_trees << " trees of depth "
            << tree_options.max_depth << " with min sample " << tree_options.min_sample_count << std::endl;
        // for (datapoint_idx_t d = 0; d < num_datapoints; d++) {
        //     std::cout << d << " " << features.row(d) << std::endl;
        // }

        std::cout << "#0, 0: " << features.coeff(0, 0) << " @ " << &features.coeff(0, 0) << std::endl;

        forest_stats.label_dimensions = label_dimensions;
        forest_stats.data_dimensions = data_dimensions;
        forest_stats.num_training_datapoints = num_datapoints;

        trees.reset(new RegressionTree<FeatT, LabT, SplitT, SplFitterT>[forest_options.max_num_trees]);
        forest_stats.num_trees = forest_options.max_num_trees;
        std::cout << "created " << forest_stats.num_trees << " trees" << std::endl;

#ifdef GARF_PARALLELIZE_TBB
        std::cout << "training using TBB" << std::endl;
        // FIXME! Work out how to actually work out the number of
        // threads TBB will use, rather than guess
        parallel_for(blocked_range<tree_idx_t>(0, forest_options.max_num_trees, 5),
                     concurrent_tree_trainer<FeatT, LabT, SplitT, SplFitterT>(trees, features, labels, *this));
#else
        // Create a RNG which we will need for picking the bagging indices, plus the uniform distribution
        std::mt19937_64 rng; // Mersenne twister
        std::uniform_int_distribution<datapoint_idx_t> bagging_index_picker(0, num_datapoints - 1);

        for (tree_idx_t t = 0; t < forest_options.max_num_trees; t++) {
            trees[t].tree_id = t;

            data_indices_vec data_indices(num_datapoints);
            if (forest_options.bagging) {
                // Sample indices from the full dataset WITH REPLACEMENT!!!
                for (datapoint_idx_t d = 0; d < num_datapoints; d++) {
                    data_indices(d) = bagging_index_picker(rng);
                }
            } else {
                // gives us a vector [0, 1, 2, 3, ... num_data_points-1]
                data_indices.setLinSpaced(num_datapoints, 0, num_datapoints - 1);
            }

            // If we build the splfitter here we open the possibility of each thread building their own only
            // once, avoiding repeated memory allocation. yay!
            SplFitterT<FeatT, LabT> fitter(split_options, forest_stats.num_training_datapoints,
                                           forest_stats.data_dimensions, forest_stats.label_dimensions, cout_mutex, t);
            trees[t].train(features, labels, data_indices, tree_options, &fitter);
        }
#endif
        // We are done, so set the forest as trained
        trained = true;
    }

    // Clears everything in the forest, ie forgets all the training
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::clear() {
        std::cout << "clearing forest of " << forest_stats.num_trees << " trees." << std::endl;
        trees.reset();
        forest_stats.num_trees = 0;
        trained = false;
    }

    // Given a single feature vector, send it down each tree in turn and fill the given scoped 
    // array with pointers to which node it lands at in each
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::predict_single_vector(const feature_vec<FeatT> & feature_vec,
                                                                                  boost::scoped_array<RegressionNode<FeatT, LabT, SplitT, SplFitterT> const *> * leaf_nodes_reached) const {
        for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
            (*leaf_nodes_reached)[t] = &trees[t].evaluate(feature_vec, predict_options);
        }
    }

    // Checks dimensions of labels_out matrix. Throws an error if it is not present or wrong shape.
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::check_label_output_matrix(label_mtx<LabT> * const labels_out,
                                                                                      feat_idx_t num_datapoints_to_predict) const {
        if (labels_out == NULL) {
            throw std::invalid_argument("predict(): label ouput vector must be supplied!");
        } else if (labels_out->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): labels_out->cols() != trained label dimensions");
        } else if (labels_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): labels_out->rows() doesn't match num features to predict");
        }
    }


    // Returns false if the variances_out matrix is not present (ie we shouldn't bother computing variance),
    // true if it is present and the right shape. Throws a descriptive exception if it is present but the wrong shape
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    bool RegressionForest<FeatT, LabT, SplitT, SplFitterT>::check_variance_output_matrix(label_mtx<LabT> * const variances_out,
                                                                                         feat_idx_t num_datapoints_to_predict) const {
        if (variances_out == NULL) {
            return false;  // caller of predict() hasn't supplied a vector output, so don't compute variances
        } else if (variances_out->cols() != forest_stats.label_dimensions) {
            throw std::invalid_argument("predict(): variances_out->cols() != trained label dimensions");
        } else if (variances_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): variances_out->rows() doesn't match num features to predict");
        }
        return true; // all conditions satisfied, so we should return variances
    }

    // As above, returns true if the leaf index output matrix is the right shape
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    bool RegressionForest<FeatT, LabT, SplitT, SplFitterT>::check_leaf_index_output_matrix(tree_idx_mtx * const leaf_indices_out,
                                                                                           feat_idx_t num_datapoints_to_predict) const {
        if (leaf_indices_out == NULL) {
            return false; // we don't need to compute / return leaf indices
        } else if (leaf_indices_out->cols() != forest_stats.num_trees) {
            throw std::invalid_argument("predict(): leaf_indices_out->cols() != num_trees");
        } else if (leaf_indices_out->rows() != num_datapoints_to_predict) {
            throw std::invalid_argument("predict(): leaf_indices_out->rows() != num_datapoints_to_predict");
        }
        return true; // all conditions satisfied, okay to return leaf indices
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::predict(const feature_mtx<FeatT> & features,
                                                                    label_mtx<LabT> * const labels_out,
                                                                    label_mtx<LabT> * const variances_out,
                                                                    tree_idx_mtx * const leaf_indices_out) const {
        if (!trained) {
            throw std::invalid_argument("cannot predict, forest not trained yet");
        }

        // Check features
        feat_idx_t num_datapoints_to_predict = features.rows();
        if (features.cols() != forest_stats.data_dimensions) {
            throw std::invalid_argument("predict(): feature_mtx<FeatT>.cols() != trained data dimensions");
        }

        // Don't need to get a boolean variable back from the label_output test function - if it's invalid
        // we simply throw an exception and bail out
        check_label_output_matrix(labels_out, num_datapoints_to_predict);
        bool outputting_variances = check_variance_output_matrix(variances_out, num_datapoints_to_predict);
        bool outputting_leaf_indices = check_leaf_index_output_matrix(leaf_indices_out, num_datapoints_to_predict);

        std::cout << "in predict(), tests passed" << std::endl;

        // Clear the outputs as we will sum into them
        labels_out->setZero();
        if (outputting_variances) {
            variances_out->setZero();
        }
        if (outputting_leaf_indices) {
            leaf_indices_out->setZero();
        }

        // scoped array so we are exception safe. This array contains pointers to const RegressionNodes, so
        // we can't change the node in any way
        boost::scoped_array<RegressionNode<FeatT, LabT, SplitT, SplFitterT> const *> leaf_nodes_reached;
        leaf_nodes_reached.reset(new RegressionNode<FeatT, LabT, SplitT, SplFitterT> const *[forest_stats.num_trees]);

        // NB it kind of sucks to test whether we are doing variance & leaf index outputting on every iteration
        // of the for loop. However to do the tests outside the for loop we'd need 4 different for loops doing every combination
        // of variance yes/no, leaf index yes/no - seems like a lot of code repetition which is going to be hellish for 
        // maintenance. For now I will leave it as is, but this is a FIXME in case prediction performance becomes a bottleneck.
        // I have chosen to do it this way as it means we only need to do the actual predictions - working out which leaf
        // node a particular datapoint lands at - the minimum number of times.
        for (feat_idx_t feat_vec_idx = 0; feat_vec_idx < num_datapoints_to_predict; feat_vec_idx++) {
#ifdef VERBOSE
            std::cout << "predicting on datapoint #" << feat_vec_idx << ": " << features.row(feat_vec_idx) << std::endl;
#endif
            // for each datapoint, we want to work out the set of leaf nodes it reaches, 
            // then worry about whether we are calculating variances or whatever else. We fill our scoped_array
            // with pointers to the leaf node reached by each datapoint
            predict_single_vector(features.row(feat_vec_idx), &leaf_nodes_reached);


            // This if test and the one below are suboptimal, see explanation at top of for loop
            if (!outputting_variances) {
                // Calculate mean only - simplest case, using naive method
                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    labels_out->row(feat_vec_idx) += leaf_nodes_reached[t]->dist.mean;
                }
                labels_out->row(feat_vec_idx) /= forest_stats.num_trees;
            } else {
                // Compute mean and variance at the same time. We are using iterative method for calculating each
                // online (this is most numerically stable) - see http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
                label_vec<LabT> mu_n(forest_stats.label_dimensions);
                label_vec<LabT> mu_n_minus_1(forest_stats.label_dimensions);  // mean at previous timestep
                mu_n.setZero();
                mu_n_minus_1.setZero();

                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    // Get const reference to the mean at the leaf node, just so we don't have to do the pointer indirections again
                    const feature_vec<FeatT> & leaf_node_mean = leaf_nodes_reached[t]->dist.mean;

                    // Update the mean
                    mu_n = mu_n_minus_1 + (1.0 / static_cast<double>(t+1)) * (leaf_node_mean - mu_n_minus_1);
                    // sum_x_sq += leaf_node_mean.cWiseProduct(leaf_node_mean);
                    mu_n_minus_1 = mu_n;
                    variances_out->row(feat_vec_idx) += (leaf_node_mean - mu_n_minus_1).cwiseProduct(leaf_node_mean - mu_n);
                }

                // FIXME: swap the two lines below when I have a version of clang++ with the bug fixed
                // labels_out->row(feat_vec_idx) = mu_n;
                labels_out->row(feat_vec_idx).operator=(mu_n);

                // Need this division since the calculation above computes S = num_datapoints * variance.
                // After this division, we just have the variance which is what we want.
                variances_out->row(feat_vec_idx) /= static_cast<double>(forest_stats.num_trees);
            }

            if (outputting_leaf_indices) {
                for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
                    leaf_indices_out->coeffRef(feat_vec_idx, t) = leaf_nodes_reached[t]->node_id;
                }
            }
        }
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    inline std::ostream& operator<< (std::ostream& stream, const RegressionForest<FeatT, LabT, SplitT, SplFitterT> & frst) {
        stream << "[RegFrst:";
        if (!frst.trained) {
            stream << "<not trained>]";
            return stream;
        }

        stream << frst.forest_stats.num_trees << " trees]";
        return stream;
    }

#ifdef GARF_PYTHON_BINDINGS_ENABLE

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::py_train(PyObject * features_np,
                                                                     PyObject * labels_np) {
        // copy Numpy format into eigen format
        boost::shared_ptr<const feature_mtx<FeatT> > features(numpy_obj_to_eigen_copy<FeatT>(features_np));
        std::cout << "after conversion into eigen:" << std::endl
            << "features.shape = (" << features->rows() << "," << features->cols()
            << "), contents = " << std::endl << *features << std::endl;

        boost::shared_ptr<const label_mtx<LabT> > labels(numpy_obj_to_eigen_copy<LabT>(labels_np));
        std::cout << "labels.shape = (" << labels->rows() << "," << labels->cols()
            << "), contents = " << std::endl << *labels << std::endl;

        train(*features, *labels);
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::py_predict_mean(PyObject * features_np,
                                                                            PyObject * predict_mean_out_np) {

        // Convert features into python format
        boost::shared_ptr<const feature_mtx<FeatT> > features(numpy_obj_to_eigen_copy<FeatT>(features_np));
        eigen_idx_t num_datapoints = features->rows();

        // Create a temporary eigen array to get the results, which we will copy to a numpy output matrix
        label_mtx<LabT> predict_mean_out_eig(num_datapoints, forest_stats.label_dimensions);

        // Make the call to the rest of the forest (this does proper error checking of the sizes, etc)
        // then copy the answers into the numpy output variable
        predict(*features, &predict_mean_out_eig);
        copy_eigen_data_to_numpy<LabT>(predict_mean_out_eig, predict_mean_out_np);
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::py_predict_mean_var(PyObject * features_np, 
                                                                       PyObject * predict_mean_out_np,
                                                                       PyObject * predict_var_out_np) const {

        // Convert features into python format
        boost::shared_ptr<const feature_mtx<FeatT> > features(numpy_obj_to_eigen_copy<FeatT>(features_np));
        eigen_idx_t num_datapoints = features->rows();

        // create temporary eigen arrays
        label_mtx<LabT> predict_mean_out_eig(num_datapoints, forest_stats.label_dimensions);
        label_mtx<LabT> predict_var_out_eig(num_datapoints, forest_stats.label_dimensions);

        // do the prediction, copy data to numpy outputs
        predict(*features, &predict_mean_out_eig, &predict_var_out_eig);
        copy_eigen_data_to_numpy<LabT>(predict_mean_out_eig, predict_mean_out_np);
        copy_eigen_data_to_numpy<LabT>(predict_var_out_eig, predict_var_out_np);
    }

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    void RegressionForest<FeatT, LabT, SplitT, SplFitterT>::py_predict_mean_var_leaves(PyObject * features_np,
                                                                       PyObject * predict_mean_out_np,
                                                                       PyObject * predict_var_out_np,
                                                                       PyObject * leaf_indices_out_np) const {


        // Convert features into python format
        boost::shared_ptr<const feature_mtx<FeatT> > features(numpy_obj_to_eigen_copy<FeatT>(features_np));
        eigen_idx_t num_datapoints = features->rows();

        // create temporary eigen arrays
        label_mtx<LabT> predict_mean_out_eig(num_datapoints, forest_stats.label_dimensions);
        label_mtx<LabT> predict_var_out_eig(num_datapoints, forest_stats.label_dimensions);
        tree_idx_mtx leaf_indices_out_eig(num_datapoints, forest_stats.num_trees);

        // do the prediction, copy data to numpy outputs
        predict(*features, &predict_mean_out_eig, &predict_var_out_eig, &leaf_indices_out_eig);
        copy_eigen_data_to_numpy<LabT>(predict_mean_out_eig, predict_mean_out_np);
        copy_eigen_data_to_numpy<LabT>(predict_var_out_eig, predict_var_out_np);
        copy_eigen_data_to_numpy<tree_idx_t>(leaf_indices_out_eig, leaf_indices_out_np);
    }
#endif
}