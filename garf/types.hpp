#ifndef GARF_TYPES_HPP
#define GARF_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
// #include <Eigen/VectorwiseOp.h>


using Eigen::Matrix;
// using Eigen::Vector;
// using Eigen::ColXpr;
using Eigen::RowVectorXd;
// using Eigen::ColVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

namespace garf {

    typedef Eigen::MatrixXd feature_matrix;
    typedef Eigen::VectorXd feature_vector;
    typedef Eigen::MatrixXd label_matrix;
    typedef Eigen::VectorXi indices_vector;
    typedef uint32_t tree_idx_t;
    typedef uint64_t node_idx_t;
    typedef uint32_t label_idx_t;
    typedef uint32_t feat_idx_t;
    typedef uint32_t depth_idx_t;
    typedef enum { LEFT, RIGHT } split_dir_t;

}

#endif