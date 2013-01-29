#ifndef GARF_REGRESSION_FOREST_HPP
#define GARF_REGRESSION_FOREST_HPP

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "types.hpp"
#include "util/multi_dim_gaussian.hpp"

namespace garf {
    /* Options which work at the top level of training - how
      many trees to make, what data to give them etc */
    struct ForestOptions {
        uint32_t max_num_trees;
        bool bagging;
        ForestOptions() : max_num_trees(10), bagging(false) {}
    };

    /* Options which are needed inside a tree - ie when to stop splitting.
      These are needed regardless of how we are doing the splitting */
    struct TreeOptions {
        uint32_t max_depth;
        uint32_t min_sample_count; // don't bother with a split if
        double min_variance;
        TreeOptions() : min_sample_count(10), min_variance(0.00001) {}
    };

    /* Options for how to split the tree */
    struct SplitOptions {

    };

    /* Options for how we do the prediction (early stopping at maximum depth for example) */
    struct PredictOptions {
        depth_t maximum_depth;
    };

    /* This is stuff we don't specify in advance, but we can query any forest about it. */
    struct ForestStats {
        uint32_t data_dimensions;
        uint32_t label_dimensions;
        uint32_t num_training_datapoints;
    };

    template<class SplitT>
    class RegressionNode {
        // parent node and children (left and right) nodes. Use a shared_ptr
        // so that when this node is destructed it also destructs the children
        // automatically
        const RegressionNode<SplitT> * const parent;
        boost::shared_ptr<RegressionNode<SplitT> > left;
        boost::shared_ptr<RegressionNode<SplitT> > right;

        // Label distribution
        MultiDimGaussianX dist;

        // Which training data points passed through here
        indices_vector training_data_indices;

        // Keep track of our identity and place in tree
        const node_id_t node_id;
        const depth_t depth;

        // The split object. This just holds the raw data necessary for
        // splitting - all intermediate data used while training should
        // be gone at test time leaving just the essentials
        SplitT split;
    public:
        // When constructing, the only thing we should need to store is the parent (this
        // is allowed to be null, when we have the root node
        RegressionNode(node_id_t _node_id,
                       const RegressionNode<SplitT> * const _parent,
                       label_idx_t _num_label_dims, depth_t _depth)
            : parent(_parent), dist(_num_label_dims), node_id(_node_id), depth(_depth) {};
        inline ~RegressionNode() {};

        inline void train() { LOG(INFO) << "decoy train()" << std::endl; }

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


    template<class SplitT>
    class RegressionTree {
        boost::shared_ptr<RegressionNode<SplitT> > root;
    public:
        tree_id_t tree_id;
        void train(const feature_matrix & features,
                   const label_matrix & labels,
                   const indices_vector & data_indices,
                   const TreeOptions & tree_opts,
                   const SplitOptions & split_opts);

        // Given some data vector, return a const reference to the node it would stop at
        const RegressionNode<SplitT> & evaluate(const feature_vector & fvec, const PredictOptions & predict_opts);
    };

       
    // We don't want to template on feature or label
    // type any more. Features and labels are
    // doubles. What we do want to template on is
    // the type of feature splitter we have, but let's get the
    // main interface down for now 
    template<class SplitT>
    class RegressionForest {
        bool is_trained;

        ForestOptions forest_options;
        TreeOptions tree_options;
        SplitOptions split_options;

        ForestStats forest_stats;

        boost::shared_array<RegressionTree<SplitT> > trees;
    public:
        RegressionForest() : is_trained(false)  {};
        inline ~RegressionForest() {}

        // Train a forest with some dataset
        inline void train() { std::cout << "blah" << std::endl; }
        void train(const feature_matrix & features, const label_matrix & labels);

        void predict(const feature_matrix & features, label_matrix * output_labels, label_matrix * output_variance);
    };

    



}

// Need to include all the implementation here or else the compiler can't access
// the declarations (compiling the .cpp files produces object files which don't 
// really have anything inside).
#include "regression_forest.cpp"
#include "regression_tree.cpp"
#include "regression_node.cpp"


#endif