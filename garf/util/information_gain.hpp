#ifndef GARF_UTIL_INFORMATION_GAIN_HPP
#define GARF_UTIL_INFORMATION_GAIN_HPP

#include <cmath>
#include <Eigen/LU>

namespace garf {

    template<typename T>
    double information_gain(const MultiDimGaussianX<T> & parent_dist,
                            const MultiDimGaussianX<T> & left_dist,
                            const MultiDimGaussianX<T> & right_dist,
                            uint64_t num_in_parent, uint64_t num_in_left, uint64_t num_in_right) {
        double det_parent = parent_dist.cov.determinant();
        double det_left = left_dist.cov.determinant();
        double det_right = right_dist.cov.determinant();

        double log_det_parent = log(det_parent);
        double log_det_left = log(det_left);
        double log_det_right = log(det_right);

        double inf_gain = log_det_parent;
        inf_gain -= ((num_in_left * log_det_left) / static_cast<double>(num_in_parent));
        inf_gain -= ((num_in_right * log_det_right) / static_cast<double>(num_in_parent));

        if (inf_gain == std::numeric_limits<double>::infinity()) {
#ifdef VERBOSE
            // LOG(ERROR)
            std::cout << std::endl << "infinite information gain from " << std::endl 
                << "parent:" << num_in_parent << ":" << parent_dist << std::endl
                << "left:" << num_in_left << ":" << left_dist << std::endl
                << "right:" << num_in_right << ":" << right_dist << std::endl
                << "is being replaced with -Inf" << std::endl;
#endif
            inf_gain = -std::numeric_limits<double>::infinity();
        }

        return inf_gain;
    }
}


#endif