#ifndef GARF_TYPES_HPP
#define GARF_TYPES_HPP

// #include <Eigen/Dense>
#include <Eigen/Core>
// #include <Eigen/VectorwiseOp.h>


using Eigen::Matrix;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

namespace garf {

    // 64 bit for everything. This will probably suck on 32 bit machines, but whatever
    typedef long eigen_idx_t;
    typedef double feat_t;
    typedef double label_t;
    typedef eigen_idx_t tree_idx_t;
    typedef eigen_idx_t node_idx_t;
    typedef eigen_idx_t label_idx_t;
    typedef eigen_idx_t feat_idx_t;
    typedef eigen_idx_t depth_idx_t;
    typedef eigen_idx_t data_dim_idx_t;
    typedef eigen_idx_t datapoint_idx_t;
    typedef eigen_idx_t split_idx_t;
    typedef enum { LEFT=0, RIGHT=1 } split_dir_t;

    typedef Eigen::MatrixXd feature_matrix;
    typedef Eigen::VectorXd feature_vector;
    typedef Eigen::MatrixXd label_matrix;
    typedef Eigen::MatrixXd variance_matrix;
    typedef Eigen::VectorXi indices_vector;
    typedef Eigen::Matrix<node_idx_t, Eigen::Dynamic, Eigen::Dynamic> tree_idx_matrix;
    typedef Eigen::Matrix<label_t, Eigen::Dynamic, 1> label_vector;
    typedef Eigen::Matrix<feat_idx_t, Eigen::Dynamic, 1> feat_idx_vector;
    typedef Eigen::Matrix<split_dir_t, Eigen::Dynamic, 1> split_dir_vector;


}

#endif