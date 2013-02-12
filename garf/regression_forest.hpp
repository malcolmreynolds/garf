#ifndef GARF_REGRESSION_FOREST_HPP
#define GARF_REGRESSION_FOREST_HPP

#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "types.hpp"
#include "options.hpp"
#include "splitter.hpp"
#include "split_fitter.hpp"
#include "util/multi_dim_gaussian.hpp"


namespace garf {

    // This is stuff about a forest we don't specify in advance,
    //but once a forest is trained we can query it..
    struct ForestStats {
        data_dim_idx_t data_dimensions;
        label_idx_t label_dimensions;
        datapoint_idx_t num_training_datapoints;
        tree_idx_t num_trees;
        inline ForestStats() : data_dimensions(-1), label_dimensions(-1), num_training_datapoints(-1), num_trees(0) {}
#ifdef GARF_SERIALIZE_ENABLE
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
#endif
    };

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT> class RegressionTree;
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>class RegressionNode;
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>class RegressionForest;

    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    class RegressionNode {
    public:
        // parent node and children (left and right) nodes. Use a shared_ptr
        // so that when this node is destructed it also destructs the children
        // automatically
        const RegressionNode<FeatT, LabT, SplitT, SplFitterT> * const parent;
        boost::shared_ptr<RegressionNode<FeatT, LabT, SplitT, SplFitterT> > left;
        boost::shared_ptr<RegressionNode<FeatT, LabT, SplitT, SplFitterT> > right;

        // Keep track of our identity and place in tree
        const node_idx_t node_id;
        const depth_idx_t depth;

        // distribution over label values
        MultiDimGaussianX<LabT> dist;

        // Which training data points passed through here
        data_indices_vec training_data_indices;

        // The split object. This just holds the raw data necessary for
        // splitting - all intermediate data used while training should
        // be gone at test time leaving just the essentials
        SplitT<FeatT> split;

        // This kind of sucks but to serialize we need a flag as to whether we 
        // save & load child nodes - easy when saving, but when loading how
        // can we know?
        bool is_leaf;

        // When constructing, the only thing we should need to store is the parent (this
        // is allowed to be null, when we have the root node
        RegressionNode(node_idx_t _node_id,
                       const RegressionNode<FeatT, LabT, SplitT, SplFitterT> * const _parent,
                       label_idx_t _num_label_dims, depth_idx_t _depth)
            : parent(_parent), node_id(_node_id), depth(_depth), dist(_num_label_dims), is_leaf(true) {};
        inline ~RegressionNode() {};

        inline void train() { std::cout << "decoy train()" << std::endl; }

        // First 5 arguments all compulsory. Last one allows us to optionally
        // provide an initial distribution, which otherwise we will need to calculate
        void train(const RegressionTree<FeatT, LabT, SplitT, SplFitterT> & tree,
                   const feature_mtx<FeatT> & features,
                   const label_mtx<LabT> & labels,
                   const data_indices_vec & data_indices,
                   const TreeOptions & tree_opts,
                   SplFitterT<FeatT, LabT> * fitter,
                   const MultiDimGaussianX<LabT> * const _dist = NULL);

        // decides whether the datapoints that reach this node justify further splitting
        bool stopping_conditions_reached(const TreeOptions & tree_opts) const;

        // Small utility functions
        inline uint32_t num_training_datapoints() const { return training_data_indices.size(); }
        inline node_idx_t left_child_index() const { return (2 * node_id) + 1; }
        inline node_idx_t right_child_index() const { return (2 * node_id) + 2; }

        template<typename F, typename L, template<typename> class S, template<typename,typename> class ST>
        friend std::ostream& operator<< (std::ostream& stream, const RegressionNode<F, L, S, ST> & node);

#ifdef GARF_SERIALIZE_ENABLE
        // Zero arg constructor just for serialization of things inside a shared_ptr
        inline RegressionNode() : parent(NULL), node_id(-1), depth(-1), dist(0) {}
    private:
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const;

        template<class Archive>
        void load(Archive & ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER();
#endif
    };


    // template<A, B>
    // BOOST_SERIALIZATION_SHARED_PTR(RegressionNode)


    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    class RegressionTree {
        boost::shared_ptr<RegressionNode<FeatT, LabT, SplitT, SplFitterT> > root;
    public:
        tree_idx_t tree_id;
        void train(const feature_mtx<FeatT> & features,
                   const label_mtx<LabT> & labels,
                   const data_indices_vec & data_indices,
                   const TreeOptions & tree_opts,
                   SplFitterT<FeatT, LabT> * fitter);

        // Given some data vector, return a const reference to the node it would stop at
        const RegressionNode<FeatT, LabT, SplitT, SplFitterT> & evaluate(const feature_vec<FeatT> & fvec,
                                                                         const PredictOptions & predict_options) const;

        template<typename F, typename L, template<typename> class S, template<typename,typename> class ST>
        friend std::ostream& operator<< (std::ostream& stream, const RegressionTree<F, L, S, ST> & tree);

#ifdef GARF_SERIALIZE_ENABLE
    private:
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
#endif
    };

       
    // We don't want to template on feature or label
    // type any more. Features and labels are
    // doubles. What we do want to template on is
    // the type of feature splitter we have, but let's get the
    // main interface down for now 
    template<typename FeatT, typename LabT, template<typename> class SplitT, template<typename, typename> class SplFitterT>
    class RegressionForest {
        bool trained;

        ForestStats forest_stats;

        boost::shared_array<RegressionTree<FeatT, LabT, SplitT, SplFitterT> > trees;

        // Checking the size of inputs given during prediction
        void check_label_output_matrix(label_mtx<LabT> * const labels_out, datapoint_idx_t num_datapoints_to_predict) const;
        bool check_variance_output_matrix(label_mtx<LabT> * const variances_out, datapoint_idx_t num_datapoints_to_predict) const;
        bool check_leaf_index_output_matrix(tree_idx_mtx * const leaf_indices_out, datapoint_idx_t num_datapoints_to_predict) const;
        void predict_single_vector(const feature_vec<FeatT> & feature_vec,
                                   boost::scoped_array<RegressionNode<FeatT, LabT, SplitT, SplFitterT> const *> * leaf_nodes_reached) const;


    public:

        ForestOptions forest_options;
        TreeOptions tree_options;
        SplitOptions split_options;
        PredictOptions predict_options;

        RegressionForest() : trained(false), forest_options(), tree_options(), split_options(), predict_options()  {};
        inline ~RegressionForest() {}

        // Below here is the main public API for interacting with forests
        void clear();
        void train(const feature_mtx<FeatT> & features, const label_mtx<LabT> & labels);
        void predict(const feature_mtx<FeatT> & features, label_mtx<LabT> * const labels_out,
                     label_mtx<LabT> * const variances_out = NULL, tree_idx_mtx * const leaf_indices_output = NULL) const;
        inline bool is_trained() const { return trained; }

        // Need different template parameters here to avoid shadowing the ones for the whole class
        template<typename F, typename L, template<typename> class S, template<typename,typename> class ST>
        friend std::ostream& operator<< (std::ostream& stream, const RegressionForest<F, L, S, ST> & frst);

        // Get a const reference to the forest stats, so they can't be changed
        inline const ForestStats & stats() const { return forest_stats; }

#ifdef GARF_SERIALIZE_ENABLE
        void save_forest(std::string filename) const;
        void load_forest(std::string filename);
        RegressionForest(std::string filename);
    private:
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const;

        template<class Archive>
        void load(Archive & ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER();
#endif
    };
}

// Need to include all the implementation here or else the compiler can't access
// the declarations (compiling the .cpp files produces object files which don't 
// really have anything inside).
#include "regression_forest.cpp"
#include "regression_tree.cpp"
#include "regression_node.cpp"

// Contains all the serialization
#include "serialization.cpp"


#endif