#ifndef GARF_REGRESSION_FOREST_HPP
#define GARF_REGRESSION_FOREST_HPP

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>


#include "util/multi_dim_gaussian.hpp"

namespace garf {

    typedef Eigen::MatrixXd feature_matrix;
    typedef Eigen::MatrixXd label_matrix;
    typedef Eigen::VectorXi indices_vector;
    typedef uint32_t tree_id_t;
    typedef uint32_t node_id_t;
    typedef uint32_t label_idx_t;

    /* Options which work at the top level of training - how
      many trees to make, what data to give them etc */
    class ForestOptions {
    public:
        uint32_t max_num_trees;
        bool bagging;
        ForestOptions() : max_num_trees(10), bagging(false) {}
    };

    /* Options which are needed inside a tree - ie when to stop splitting.
      These are needed regardless of how we are doing the splitting */
    class TreeOptions {
    public:
        uint32_t max_depth;
        uint32_t min_sample_count; // don't bother with a split if
        double min_variance;
        TreeOptions() : min_sample_count(10), min_variance(0.00001) {}
    };

    /* Options for how to split the tree */
    class SplitOptions {

    };

    /* This is stuff we don't specify in advance, but we can query any forest about it. */
    class ForestStats {
    public:
        uint32_t data_dimensions;
        uint32_t label_dimensions;
        uint32_t num_training_datapoints;
    };

    class RegressionNode {
        // parent node and children (left and right) nodes. Use a shared_ptr
        // so that when this node is destructed it also destructs the children
        // automatically
        const RegressionNode * const parent;
        boost::shared_ptr<RegressionNode> left;
        boost::shared_ptr<RegressionNode> right;

        // Label distribution
        MultiDimGaussianX dist;

        // Which training data points passed through here
        indices_vector training_data_indices;

        // Keep track of our identity and place in tree
        const node_id_t node_id;
        const uint32_t depth;

    public:
        // When constructing, the only thing we should need to store is the parent (this
        // is allowed to be null, when we have the root node
        RegressionNode(node_id_t _node_id,
                       const RegressionNode * const _parent,
                       label_idx_t _num_label_dims, uint32_t _depth)
            : parent(_parent), dist(_num_label_dims), node_id(_node_id), depth(_depth) {};
        inline ~RegressionNode() {};

        // First 5 arguments all compulsory. Last one allows us to optionally
        // provide an initial distribution, which otherwise we will need to calculate
        void train(const feature_matrix & features,
                   const label_matrix & labels,
                   const indices_vector & data_indices,
                   const TreeOptions & tree_opts,
                   const SplitOptions & split_opts,
                   const MultiDimGaussianX * const _dist = NULL);

        // decides whether the datapoints that reach this node justify further splitting
        bool stopping_conditions_reached(const TreeOptions & tree_opts) const;

        // Small utility functions
        inline uint32_t num_training_datapoints() const { return training_data_indices.size(); }
    };


    class RegressionTree {
        boost::shared_ptr<RegressionNode> root;
    public:
        tree_id_t tree_id;
        void train(const feature_matrix & features,
                   const label_matrix & labels,
                   const indices_vector & data_indices,
                   const TreeOptions & tree_opts,
                   const SplitOptions & split_opts);
    };

       
    // We don't want to template on feature or label
    // type any more. Features and labels are
    // doubles. What we do want to template on is
    // the type of feature splitter we have, but let's get the
    // main interface down for now 
    class RegressionForest {
        bool is_trained;

        ForestOptions forest_options;
        TreeOptions tree_options;
        SplitOptions split_options;

        ForestStats forest_stats;

        boost::shared_array<RegressionTree> trees;
    public:
        RegressionForest() : is_trained(false)  {};
        inline ~RegressionForest() {}

        // Train a forest with some dataset
        void train(const feature_matrix & features, const label_matrix & labels);


        void predict(const feature_matrix & features, label_matrix * output_labels, label_matrix * output_variance);
    };

    



}

#endif