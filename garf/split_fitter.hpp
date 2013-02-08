#ifndef GARF_SPLIT_FITTER_HPP
#define GARF_SPLIT_FITTER_HPP

#include <random>

#include "options.hpp"
#include "util/multi_dim_gaussian.hpp"
#include "util/information_gain.hpp"

namespace garf {

    class SplFitter {
    public:
        const SplitOptions split_opts;
        const uint64_t label_dims;

        MultiDimGaussianX left_child_dist;
        MultiDimGaussianX right_child_dist;

        std::mt19937_64 rng; // Mersenne twister

        SplFitter(const SplitOptions & _split_opts, uint64_t _label_dims, uint64_t seed_value=42)
                : split_opts(_split_opts),
                  label_dims(_label_dims), 
                  left_child_dist(_label_dims),
                  right_child_dist(_label_dims) {
            std::cout << "seeding RNG with " << seed_value << std::endl;
            rng.seed(seed_value);
        };
    };

    class AxisAlignedSplFitter : public SplFitter {
    public:
        AxisAlignedSplFitter(const SplitOptions & _split_opts, uint64_t _label_dims, uint64_t seed_value=0) : SplFitter(_split_opts, _label_dims, seed_value) {};
        virtual bool choose_split_parameters(const feature_matrix & features,
                                             const feature_matrix & labels,
                                             const indices_vector & parent_data_indices,
                                             const MultiDimGaussianX & parent_dist,
                                             AxisAlignedSplt * split,
                                             indices_vector * left_child_indices_out,
                                             indices_vector * right_child_indices_out);
    };
}

#include "impl/axis_aligned_splfitter.cpp"

#endif