#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

using namespace boost::python;

//#define BUILD_PYTHON_BINDINGS

// This hopefully means doing a release compile with bjam means no excessive output
#ifndef NDEBUG
#define VERBOSE
#endif

#define GARF_SERIALIZE_ENABLE
#define GARF_PYTHON_BINDINGS_ENABLE
#define GARF_PARALLELIZE_TBB
#define GARF_FEATURE_IMPORTANCE

#include "garf/options.hpp"
#include "garf/regression_forest.hpp"

using namespace garf;


BOOST_PYTHON_MODULE(_garf) {

    import_array();

    class_<ForestOptions>("ForestOptions")
        .def_readwrite("max_num_trees", &ForestOptions::max_num_trees)
        .def_readwrite("bagging", &ForestOptions::bagging);

    class_<TreeOptions>("TreeOptions")
        .def_readwrite("max_depth", &TreeOptions::max_depth)
        .def_readwrite("min_sample_count", &TreeOptions::min_sample_count)
        .def_readwrite("min_variance", &TreeOptions::min_variance);

    class_<SplitOptions>("SplitOptions")
        .def_readwrite("num_splits_to_try", &SplitOptions::num_splits_to_try)
        .def_readwrite("threshes_per_split", &SplitOptions::threshes_per_split);

    class_<PredictOptions>("PredictOptions")
        .def_readwrite("maximum_depth", &PredictOptions::maximum_depth);

    class_<ForestStats>("ForestStats")
        .def_readonly("data_dimensions", &ForestStats::data_dimensions)
        .def_readonly("label_dimensions", &ForestStats::label_dimensions)
        .def_readonly("num_training_datapoints", &ForestStats::num_training_datapoints)
        .def_readonly("num_trees", &ForestStats::num_trees);

    // The following classes must be exposed multiple times, once for each type we
    // may want to use them with. Python does not know about our C++ templates, so we
    // must explicitly expose the MultiDimGaussian over doubles, and separately from that
    // that the one over floats, etc

    #define MULTI_DIM_GAUSSIAN_BINDINGS(T, N) \
    class_<util::MultiDimGaussianX<T> >("MultiDimGaussian" N) \
        .def_readonly("dimensions", &util::MultiDimGaussianX<T>::dimensions) \
        .add_property("mean", &util::MultiDimGaussianX<T>::get_mean_numpy) \
        .add_property("cov", &util::MultiDimGaussianX<T>::get_cov_numpy)

    MULTI_DIM_GAUSSIAN_BINDINGS(double, "D");
    MULTI_DIM_GAUSSIAN_BINDINGS(float, "F");


    // First four arguments are the template parameters to the forest classes - ie feature type,
    // label type, splitter type, split finder type. Subsequent arguments are strings use to make
    // a unqiue name for the class in python - we can't have all the different instantiations of the
    // template all just called "RegressionForest" unfortunately
    #define EXPOSE_FOREST_CLASSES(F, L, S, SF, FN, LN, SN) \
    class_<RegressionForest<F, L, S, SF> >("RegressionForest" FN LN SN) \
        .add_property("trained", &RegressionForest<F, L, S, SF>::is_trained) \
        .add_property("stats", make_function(&RegressionForest<F, L, S, SF>::stats, \
                                             return_internal_reference<>())) \
        .def_readonly("forest_options", &RegressionForest<F, L, S, SF>::forest_options) \
        .def_readonly("tree_options", &RegressionForest<F, L, S, SF>::tree_options) \
        .def_readonly("split_options", &RegressionForest<F, L, S, SF>::split_options) \
        .def_readonly("predict_options", &RegressionForest<F, L, S, SF>::predict_options) \
        .def("_train", &RegressionForest<F, L, S, SF>::py_train) \
        .def("_predict", &RegressionForest<F, L, S, SF>::py_predict_mean) \
        .def("_predict", &RegressionForest<F, L, S, SF>::py_predict_mean_var) \
        .def("_predict", &RegressionForest<F, L, S, SF>::py_predict_mean_var_leaves) \
        .def("get_tree", &RegressionForest<F, L, S, SF>::get_tree, \
             return_value_policy<copy_const_reference>()) \
        .def("load_forest", &RegressionForest<F, L, S, SF>::load_forest) \
        .def("save_forest", &RegressionForest<F, L, S, SF>::save_forest); \
    class_<RegressionTree<F, L, S, SF> >("RegressionTree" FN LN SN) \
        .def_readonly("tree_id", &RegressionTree<F, L, S, SF>::tree_id) \
        .add_property("root", make_function(&RegressionTree<F, L, S, SF>::get_root, \
                                            return_value_policy<copy_const_reference>())); \
    class_<RegressionNode<F, L, S, SF> >("RegressionNode" FN LN SN) \
        .def_readonly("is_leaf", &RegressionNode<F, L, S, SF>::is_leaf) \
        .def_readonly("id", &RegressionNode<F, L, S, SF>::node_id) \
        .def_readonly("depth", &RegressionNode<F, L, S, SF>::depth) \
        .add_property("num_samples", make_function(&RegressionNode<F, L, S, SF>::num_samples)) \
        .add_property("training_indices", &RegressionNode<F, L, S, SF>::get_training_indices) \
        .add_property("dist", make_function(&RegressionNode<F, L, S, SF>::get_dist, \
                                            return_value_policy<copy_const_reference>())) \
        .add_property("l", make_function(&RegressionNode<F, L, S, SF>::get_left, \
                                         return_value_policy<copy_const_reference>())) \
        .add_property("r", make_function(&RegressionNode<F, L, S, SF>::get_right, \
                                         return_value_policy<copy_const_reference>()))

    // Operating on doubles & using axis aligned is a reasonable default, so we don't need 
    // any special strings in the class name - ie the "RegressionForest" python side class will
    // be this case of double features, double lables, axis aligned splits
    EXPOSE_FOREST_CLASSES(double, double, AxisAlignedSplt, AxisAlignedSplFitter, "_D", "_D", "");
    EXPOSE_FOREST_CLASSES(float, float, AxisAlignedSplt, AxisAlignedSplFitter, "_F", "_F", "");

}

