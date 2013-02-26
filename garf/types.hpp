#ifndef GARF_TYPES_HPP
#define GARF_TYPES_HPP

#include <random>

// #include <Eigen/Dense>
#include <Eigen/Core>

// #include <Eigen/VectorwiseOp.h>

// using namespace Eigen;

// using Eigen::Matrix;
// using Eigen::RowVectorXd;
// using Eigen::Vector3d;
// using Eigen::VectorXd;
// using Eigen::VectorXi;
// using Eigen::Matrix3d;
// using Eigen::MatrixXd;

namespace garf {

    // 64 bit for everything. This will probably suck on 32 bit machines, but whatever
    typedef long eigen_idx_t;
    typedef eigen_idx_t tree_idx_t;
    typedef eigen_idx_t node_idx_t;
    typedef eigen_idx_t label_idx_t;
    typedef eigen_idx_t feat_idx_t;
    typedef eigen_idx_t depth_idx_t;
    typedef eigen_idx_t data_dim_idx_t;
    typedef eigen_idx_t datapoint_idx_t;
    typedef eigen_idx_t split_idx_t;
    typedef double importance_t;
    typedef double error_t;
    typedef enum { LEFT=0, RIGHT=1 } split_dir_t;

    typedef std::mt19937_64 RngSource;

    // This will only compile with C++ 11. Sorry folks. There is lots of redundancy here, but
    // it makes the source code lovely to look at. Hooray.
    template <typename T> using feature_mtx = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T> using feature_vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template <typename T> using label_mtx = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T> using label_vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template <typename T> using variance_mtx = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T> using variance_vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    typedef Eigen::Matrix<datapoint_idx_t, Eigen::Dynamic, 1> data_indices_vec;
    typedef Eigen::Matrix<datapoint_idx_t, Eigen::Dynamic, Eigen::Dynamic> data_indices_mtx;

    typedef Eigen::Matrix<node_idx_t, Eigen::Dynamic, Eigen::Dynamic> tree_idx_mtx;
    typedef Eigen::Matrix<feat_idx_t, Eigen::Dynamic, 1> feat_idx_vec;
    typedef Eigen::Matrix<split_dir_t, Eigen::Dynamic, 1> split_dir_vec;
    typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> bool_vec;

    typedef Eigen::Matrix<importance_t, Eigen::Dynamic, 1> importance_vec;
    typedef Eigen::Matrix<error_t, Eigen::Dynamic, 1> error_vec;
    typedef Eigen::Matrix<error_t, Eigen::Dynamic, Eigen::Dynamic> error_mtx;
}

#endif