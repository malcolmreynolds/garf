#ifndef GARF_EIGEN_SERIALIZATION_HPP
#define GARF_EIGEN_SERIALIZATION_HPP

#include <fstream>
#include <Eigen/Core>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 

#include "types.hpp"

using namespace Eigen;

namespace boost {
    template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void serialize(Archive & ar, Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
                          const unsigned int file_version) 
    {
        garf::eigen_idx_t rows = t.rows(), cols = t.cols();
        ar & rows;
        ar & cols;
        if (rows * cols != t.size()) {
            t.resize(rows, cols);
        }

        for(garf::eigen_idx_t i=0; i<t.size(); i++) {
            ar & t.data()[i];
        }
    }
}


namespace garf {

    // Utility function which means we don't need to open an fstream, etc
    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::save_forest(std::string filename) const {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        ofs << *this;
        std::cout << "forest saved to " << filename << std::endl;
    }

    // Load a forest from disk into the forest this is called on. Note, this will
    // delete the current forest, so it makes sense to call this on a forest which
    // isn't currently trained - but there is nothing to enforce this. 
    template<class SplitT, class SplFitterT>
    void RegressionForest<SplitT, SplFitterT>::load_forest(std::string filename) {
        clear();
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ifs >> *this;
        std::cout << "forest loaded from " << filename << std::endl;
    }

    // Alternate constructor which loads from a filename straight away
    template<class SplitT, class SplFitterT>
    RegressionForest<SplitT, SplFitterT>::RegressionForest(std::string filename) {
        load_forest(filename);
    }

    // load & save foreststats
    template<class Archive>
    void ForestStats::serialize(Archive & ar, const unsigned int version) {
        ar & data_dimensions;
        ar & label_dimensions;
        ar & num_training_datapoints;
        ar & num_trees;
    }

    // save a RegressionNode
    template<class SplitT, class SplFitterT>
    template<class Archive>
    void RegressionNode<SplitT, SplFitterT>::save(Archive & ar, const unsigned int version) const {
        ar << node_id;
        ar << depth;
        ar << dist;
        ar << split;
        ar << is_leaf;
        ar << training_data_indices;
        std::cout << "saved node " << node_id << " with datapoints " << training_data_indices.transpose() << std::endl;
        // Only serialize children if there are any
        if (!is_leaf) {
            ar << left;
            ar << right;
        }

    }

    // Load a RegressionNode
    template<class SplitT, class SplFitterT>
    template<class Archive>
    void RegressionNode<SplitT, SplFitterT>::load(Archive & ar, const unsigned int version) {
        // Need to use const cast to fill in a bunch of sutff here - this
        // is necessary unfortunately, recommended by boost.serialize manual
        ar >> const_cast<node_idx_t &>(node_id);
        ar >> const_cast<depth_idx_t &>(depth);
        ar >> dist;
        ar >> split;
        ar >> is_leaf;
        ar >> training_data_indices;

        std::cout << "loaded node " << node_id << " with datapoints " << training_data_indices.transpose() << std::endl;

        if (!is_leaf) {
            ar >> left;
            ar >> right;

            // Fix the parent pointers of the loaded children. This is the recommended
            // way of setting a const variable according to the Boost.serialzie
            RegressionNode<SplitT, SplFitterT> ** left_child_parent = const_cast<RegressionNode<SplitT, SplFitterT> **>(&(left->parent));
            *left_child_parent = this;
            RegressionNode<SplitT, SplFitterT> ** right_child_parent = const_cast<RegressionNode<SplitT, SplFitterT> **>(&(right->parent));
            *right_child_parent = this;
        }
        else {
            left.reset();
            right.reset();
        }
    }

    // Save and load a RegressionTree
    template<class SplitT, class SplFitterT>
    template<class Archive>
    void RegressionTree<SplitT, SplFitterT>::serialize(Archive & ar, const unsigned int version) {
        ar & tree_id;
        ar & root;
    }

    // Save a RegressionForest
    template<class SplitT, class SplFitterT>
    template<class Archive>
    void RegressionForest<SplitT, SplFitterT>::save(Archive & ar, const unsigned int version) const {
        ar << trained;

        ar << forest_options;
        ar << tree_options;
        ar << split_options;
        ar << predict_options;
        
        ar << forest_stats;
        for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
            ar << trees[t];
        }
    }

    // Load a RegressionForest. Only complication is we need to allocate new memory in the trees array
    template<class SplitT, class SplFitterT>
    template<class Archive>
    void RegressionForest<SplitT, SplFitterT>::load(Archive & ar, const unsigned int version) {
        ar >> trained;

        ar >> forest_options;
        ar >> tree_options;
        ar >> split_options;
        ar >> predict_options;

        ar >> forest_stats;
        trees.reset(new RegressionTree<SplitT, SplFitterT>[forest_stats.num_trees]);
        for (tree_idx_t t = 0; t < forest_stats.num_trees; t++) {
            ar >> trees[t];
        }
    }

    // Load & save a Multi dimensional Gaussian distribution
    template<class Archive>
    void MultiDimGaussianX::serialize(Archive & ar, const unsigned int version) {
        ar & const_cast<eigen_idx_t &>(dimensions);
        ar & mean;
        ar & cov;
    }

    // Load & save tree options
    template<class Archive>
    void TreeOptions::serialize(Archive & ar, const unsigned int version) {
        ar & max_depth;
        ar & min_sample_count;
        ar & min_variance;
    }

    // Load & save forest options
    template<class Archive>
    void ForestOptions::serialize(Archive & ar, const unsigned int version) {
        ar & max_num_trees;
        ar & bagging;
    }

    // Load & save SplitOptions
    template<class Archive>
    void SplitOptions::serialize(Archive & ar, const unsigned int version) {
        ar & num_splits_to_try;
        ar & threshes_per_split;
    }

    // Load & save PredictOptions
    template<class Archive>
    void PredictOptions::serialize(Archive & ar, const unsigned int version) {
        ar & maximum_depth;
    }

}


#endif