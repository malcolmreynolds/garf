#ifndef GARF_SPLIT_FITTER_HPP
#define GARF_SPLIT_FITTER_HPP

#include <random>

#include "options.hpp"
#include "util/multi_dim_gaussian.hpp"
#include "util/information_gain.hpp"

namespace garf {

    template<typename FeatT, typename LabT>
    class SplFitter {
    public:
        const SplitOptions split_opts;
        const label_idx_t label_dims;
        const datapoint_idx_t total_num_datapoints;
        const feat_idx_t feature_dimensionality;

        MultiDimGaussianX<LabT> left_child_dist;
        MultiDimGaussianX<LabT> right_child_dist;

        std::mt19937_64 rng; // Mersenne twister

        // ref to the mutex which we need to lock to print anything
        tbb::mutex & print_mutex;


        SplFitter(const SplitOptions & _split_opts,
                  datapoint_idx_t _total_num_datapoints,
                  feat_idx_t _feature_dimensionality,
                  label_idx_t _label_dims,
                  tbb::mutex & _print_mutex,
                  uint64_t seed_value=42)
            : split_opts(_split_opts),
              label_dims(_label_dims), 
              total_num_datapoints(_total_num_datapoints),
              feature_dimensionality(_feature_dimensionality),
              left_child_dist(_label_dims),
              right_child_dist(_label_dims),
              print_mutex(_print_mutex) {

            // Seed the RNG
            print_mutex.lock();
            std::cout << "seeding RNG with " << seed_value << std::endl;
            print_mutex.unlock();
            rng.seed(seed_value);
        };
    };

    template<typename FeatT, typename LabT>
    class AxisAlignedSplFitter : public SplFitter<FeatT, LabT> {

        // Used to select feature indices to pick from
        std::uniform_int_distribution<feat_idx_t> feat_idx_dist;

        // Store which features we are going to look at. this is a vector
        // of length split_opts.num_splits_to_try (currently constant per node)
        feat_idx_vec feature_indices_to_evaluate;

        // Store all the different feature values for every single datapoint to land at the node.
        // We made this total_datapoints x num_splits_to_try, which will only be fully used
        // at the root node, afterwards we only use however many topmost rows as we need.
        feature_mtx<FeatT> candidate_feature_values;

        // Store the min and max values for each feature we are trying to generate data on
        feature_vec<FeatT> min_feature_values;
        feature_vec<FeatT> max_feature_values;

        // This stores the randomly samples thresholds we come up with for each feature, 
        // it is initialised to size num_splits_to_try x threshes_per_split
        feature_mtx<FeatT> split_thresholds;

        // This is used to generate the thresholds we use for splitting
        // std::uniform_real_distribution<FeatT> thresh_dist; 

        // Temporary vector to store which direction things go in (to calculate
        // information gains etc) - 
        split_dir_vec candidate_split_directions;

        // Temporary vectors to store which data is going left and right. As with feature_values
        // we only need
        data_indices_vec samples_going_left;
        data_indices_vec samples_going_right;

        // Keep track of the amounts in the above vectors which are valid
        datapoint_idx_t num_going_left;
        datapoint_idx_t num_going_right;

        LabT best_inf_gain;
        bool good_split_found;

        // Fill the feature_indices_to_evaluate vector with some new features
        void select_candidate_features();


        // For each datapoint which lands in this node
        void evaluate_datapoints_at_each_feature(const feature_mtx<FeatT> & features,
                                                 const data_indices_vec & parent_data_indices,
                                                 const datapoint_idx_t num_in_parent);

        // Find the min and max values of the features we have
        void find_min_max_features(const datapoint_idx_t num_in_parent);

        // Pick some thresholds for each candidate feature, with min and max values
        void generate_split_thresholds();

        void check_split_thresholds();

    public:

        AxisAlignedSplFitter(const SplitOptions & _split_opts,
                             datapoint_idx_t _total_num_datapoints,
                             feat_idx_t _feature_dimensionality,
                             label_idx_t _label_dims,
                             tbb::mutex & _print_mutex,
                             uint64_t seed_value)
            : SplFitter<FeatT, LabT>(_split_opts, _total_num_datapoints, _feature_dimensionality, _label_dims, _print_mutex, seed_value),
              feat_idx_dist(0, _feature_dimensionality-1),
              feature_indices_to_evaluate(_split_opts.num_splits_to_try),
              candidate_feature_values(_total_num_datapoints, _split_opts.num_splits_to_try),
              min_feature_values(_split_opts.num_splits_to_try),
              max_feature_values(_split_opts.num_splits_to_try),
              split_thresholds(_split_opts.num_splits_to_try, _split_opts.threshes_per_split),
              candidate_split_directions(_total_num_datapoints),
              samples_going_left(_total_num_datapoints),
              samples_going_right(_total_num_datapoints),
              num_going_left(-1),
              num_going_right(-1)
               {};

        void evaluate_single_split(const data_indices_vec & data_indices,
                                   const datapoint_idx_t num_in_parent,
                                   split_idx_t split_feature, FeatT thresh,
                                   split_dir_vec * candidate_split_directions,
                                   data_indices_vec * const indices_going_left,
                                   data_indices_vec * const indices_going_right,
                                   datapoint_idx_t * const num_going_left,
                                   datapoint_idx_t * const num_going_right) const;


        // This is the function we call that does everything. The return values indicates
        // whether a decent split has been found
        virtual bool choose_split_parameters(const feature_mtx<FeatT> & features,
                                             const label_mtx<LabT> & labels,
                                             const data_indices_vec & parent_data_indices,
                                             const MultiDimGaussianX<LabT> & parent_dist,
                                             AxisAlignedSplt<FeatT> * split,
                                             data_indices_vec * left_child_indices_out,
                                             data_indices_vec * right_child_indices_out);
    };
}

#include "impl/axis_aligned_splfitter.cpp"


#endif