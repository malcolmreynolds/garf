#ifndef GARF_SPLIT_FITTER_HPP
#define GARF_SPLIT_FITTER_HPP

#include "options.hpp"
#include "util/multi_dim_gaussian.hpp"

namespace garf {

    class SplFitter {
        SplitOptions split_opts;
    public:
        SplFitter(const SplitOptions & _split_opts) : split_opts(_split_opts) {};
        virtual bool choose_split_parameters(const feature_matrix & features,
                                             const feature_matrix & labels,
                                             const indices_vector & parent_data_indices,
                                             const MultiDimGaussianX & parent_dist,
                                             indices_vector * left_child_indices_out,
                                             indices_vector * right_child_indices_out) = 0;
    };

    class AxisAlignedSplFitter : public SplFitter {
    public:
        AxisAlignedSplFitter(const SplitOptions & _split_opts) : SplFitter(_split_opts) {};
        virtual bool choose_split_parameters(const feature_matrix & features,
                                             const feature_matrix & labels,
                                             const indices_vector & parent_data_indices,
                                             const MultiDimGaussianX & parent_dist,
                                             indices_vector * left_child_indices_out,
                                             indices_vector * right_child_indices_out);
    };
}

#include "impl/axis_aligned_splfitter.hpp"

#endif