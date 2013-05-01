#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

using namespace boost::python;

//#define BUILD_PYTHON_BINDINGS

// This hopefully means doing a release compile with bjam means no excessive output
#ifndef NDEBUG
#define VERBOSE
#endif

#include "decision_forest.hpp"
#include "manifold_forest.hpp"
#include "regression_forest.hpp"

using namespace garf;


BOOST_PYTHON_MODULE(_garf) {

    enum_<forest_split_type_t>("ForestSplitType")
        .value("axis_aligned", AXIS_ALIGNED)
        .value("linear_2d", LINEAR_2D)
        .value("conic", CONIC);
        
        
    enum_<split_direction_t>("SplitDirection")
        .value("left", LEFT)
        .value("right", RIGHT);
        
    enum_<affinity_distances_types_t>("AffinityDistanceType")
        .value("binary", BINARY)
        .value("gaussian", GAUSSIAN);

    enum_<single_double_split_t>("SingDoubSplitType")
        .value("single", SINGLE)
        .value("double", DOUBLE);

    enum_<split_t>("SplitType")
        .value("pair_diff", PAIR_DIFF)
        .value("pair_plus", PAIR_PLUS)
        .value("pair_abs", PAIR_ABS)
        .value("singular", SINGULAR);

#define EXPOSE_MVN(FEATT, SUFFIX) \
    class_<multivariate_normal<FEATT> >("MVNormal" SUFFIX, no_init) \
        .def(pyublas::by_value_ro_member("mu", &multivariate_normal<FEATT>::mu))  \
        .def(pyublas::by_value_ro_member("sigma", &multivariate_normal<FEATT>::sigma))
        
    EXPOSE_MVN(double, "D");
    EXPOSE_MVN(float, "F");

    class_<forest_params>("ForestParams")
        .def_readwrite("max_num_trees", &forest_params::_max_num_trees)
        .def_readwrite("max_tree_depth", &forest_params::_max_tree_depth)
        .def_readwrite("min_sample_count", &forest_params::_min_sample_count)
        .def_readwrite("num_splits_to_try", &forest_params::_num_splits_to_try)
        .def_readwrite("num_threshes_per_split", &forest_params::_num_threshes_per_split)
        .def_readwrite("bagging", &forest_params::_bagging)
        .def_readwrite("balance_bias", &forest_params::_balance_bias)
        .def_readwrite("min_variance", &forest_params::_min_variance);

    class_<manifold_forest_params, bases<forest_params> >("ManifoldForestParams");
    class_<regression_forest_params, bases<forest_params> >("RegressionForestParams");

    class_<forest_stats>("ForestStats")
        .def_readonly("num_trees", &forest_stats::_num_trees)
        .def_readonly("feature_dimensionality", &forest_stats::_feature_dimensionality)
        .def_readonly("num_training_samples", &forest_stats::_num_training_samples);
    class_<manifold_forest_stats, bases<forest_stats> >("ManifoldForestStats");
    class_<regression_forest_stats, bases<forest_stats> >("RegressionForestStats")
        .def_readonly("label_dimensionality", &regression_forest_stats::_label_dimensionality);
    
// Typing the whole thing out for every combination of feature type and label type will
// be unbearably tedious    
#define EXPOSE_SPLIT_FINDERS(FEATT, LABELT, SUFFIX)    \
    class_<split_finder<FEATT, LABELT>, boost::noncopyable>("SplitFinder" SUFFIX, no_init) \
        .add_property("num_splits_to_try", &split_finder<FEATT, LABELT>::num_splits_to_try) \
        .add_property("num_threshes_per_split", &split_finder<FEATT, LABELT>::num_threshes_per_split) \
        .def_readonly("feature_dimensionality", &split_finder<FEATT, LABELT>::_feature_dimensionality) \
        .def_readonly("inf_gain_dimensionality", &split_finder<FEATT, LABELT>::_inf_gain_dimensionality) \
        .def_readonly("num_samples", &split_finder<FEATT, LABELT>::_num_training_samples) \
        .add_property("thresh", &split_finder<FEATT, LABELT>::best_split_thresh) \
        .def("name", &split_finder<FEATT, LABELT>::name); \
    class_<axis_aligned_split_finder<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("AxisAlignedSplitFinder" SUFFIX, no_init) \
        .add_property("feature", &axis_aligned_split_finder<FEATT, LABELT>::get_split_feature); \
    class_<single_double_spltfndr<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("SingleDoubleSplitFinder" SUFFIX, no_init) \
        .add_property("type", &single_double_spltfndr<FEATT, LABELT>::get_type) \
        .add_property("feat_i", &single_double_spltfndr<FEATT, LABELT>::get_feat_i) \
        .add_property("feat_j", &single_double_spltfndr<FEATT, LABELT>::get_feat_j) \
        .add_property("coeff_i", &single_double_spltfndr<FEATT, LABELT>::get_coeff_i) \
        .add_property("coeff_j", &single_double_spltfndr<FEATT, LABELT>::get_coeff_j); \
    class_<hyperplane_split_finder<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("HyperplaneSplitFinder" SUFFIX, no_init) \
        .add_property("hyperplane_dimensionality", &hyperplane_split_finder<FEATT, LABELT>::get_hyperplane_dimensionality) \
        .add_property("plane_coeffs", &hyperplane_split_finder<FEATT, LABELT>::get_plane_coeffs) \
        .add_property("feature_subset", &hyperplane_split_finder<FEATT, LABELT>::get_feature_subset); \
    class_<smart_hyperplane_split_finder<FEATT, LABELT>, bases<hyperplane_split_finder<FEATT, LABELT> > >("SmartHyperplaneSplitFinder" SUFFIX, no_init); \
    class_<pairwise_diff_split_finder<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("PairwiseSplitFinder" SUFFIX, no_init) \
        .add_property("feat_i", &pairwise_diff_split_finder<FEATT, LABELT>::get_feat_i) \
        .add_property("feat_j", &pairwise_diff_split_finder<FEATT, LABELT>::get_feat_j); \
    class_<ctf_pairwise_only_split_finder<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("CTFPairwiseOnlySplitFinder" SUFFIX, no_init) \
        .add_property("feat_i", &ctf_pairwise_only_split_finder<FEATT, LABELT>::get_feat_i) \
        .add_property("feat_j", &ctf_pairwise_only_split_finder<FEATT, LABELT>::get_feat_j); \
    class_<ctf_pairwise_singular_split_finder<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("CTFPairwiseSingularSplitFinder" SUFFIX, no_init) \
        .add_property("type", &ctf_pairwise_singular_split_finder<FEATT, LABELT>::get_type) \
        .add_property("feat_i", &ctf_pairwise_singular_split_finder<FEATT, LABELT>::get_feat_i) \
        .add_property("feat_j", &ctf_pairwise_singular_split_finder<FEATT, LABELT>::get_feat_j); \
    class_<ncc_levels_2468_pairwise_spltfndr<FEATT, LABELT>, bases<split_finder<FEATT, LABELT> > >("NCCLev2468PrwsSplitFinder" SUFFIX, no_init) \
        .add_property("feat_i_x", &ncc_levels_2468_pairwise_spltfndr<FEATT, LABELT>::get_feat_i_x) \
        .add_property("feat_j_x", &ncc_levels_2468_pairwise_spltfndr<FEATT, LABELT>::get_feat_j_x);
    EXPOSE_SPLIT_FINDERS(double, double, "");   
    EXPOSE_SPLIT_FINDERS(float, float, "Flt");
        
//     class_<training_set<double>, boost::noncopyable>("TrainingSetDbl", no_init);
//     class_<supervised_training_set<double, double>, bases<training_set<double> >, boost::noncopyable>("SupervisedTrainingSetDbl", no_init);
//     class_<multi_supervised_regression_training_set<double, double>,
//         bases< supervised_training_set<double, double> > >("MultiSupervisedRegTrainingSetDbl",no_init);
// //        .def("hello", &multi_supervised_regression_training_set<double, double>::hello);
// //        .def(init<boost::shared_ptr<garf_types<double>::matrix>, boost::shared_ptr<garf_types<double>::matrix> >);

    class_<decision_node>("DecisionNode", no_init)
        .add_property("node_id", &decision_node::node_id)
        .add_property("num_samples", &decision_node::num_samples_landing_here)
        .add_property("samples", &decision_node::samples_landing_here) // THIS IS THE WAY TO EXPOSE POINTERS TO NUMPY STUFF
        .add_property("depth", &decision_node::depth)
        .add_property("inf_gain", &decision_node::information_gain);
    class_<decision_forest>("DecisionForest", init<>())
        .add_property("training_time", &decision_forest::training_time);
    class_<decision_tree>("DecisionTree")
        .add_property("tree_id", &decision_tree::tree_id)
        .add_property("next_free_node_id", &decision_tree::next_free_node_id);
        
        
        
// I think this is a reasonable use of the preprocessor        
#define EXPOSE_MANIFOLD_FORESTS(FEATT, SUFFIX) \
    class_<manifold_node<FEATT>, bases<decision_node> >("ManifoldNode" SUFFIX, no_init) \
        .add_property("dist", make_function(&manifold_node<FEATT>::get_sample_distribution, return_internal_reference<>())) \
        .add_property("split", make_function(&manifold_node<FEATT>::get_splitter, return_internal_reference<>())) \
        .add_property("l", make_function(&manifold_node<FEATT>::get_left, return_internal_reference<>())) \
        .add_property("r", make_function(&manifold_node<FEATT>::get_right, return_internal_reference<>())); \
    class_<manifold_tree<FEATT>, bases<decision_tree> >("ManifoldTree" SUFFIX) \
        .def("forest", &manifold_tree<FEATT>::get_forest, return_internal_reference<>()) \
        .add_property("root", make_function(&manifold_tree<FEATT>::get_root, return_internal_reference<>())); \
    class_<manifold_forest<FEATT>, bases<decision_forest> >("ManifoldForest" SUFFIX, init<>()) \
        .def(init<manifold_forest_params>()) \
		.def("hello", &manifold_forest<FEATT>::hello) \
		.def("train", &manifold_forest<FEATT>::train_py) \
		.def("train_sup", &manifold_forest<FEATT>::train_sup_py) \
		.def("compute_affinity_matrix", &manifold_forest<FEATT>::compute_affinity_matrix) \
		.def("get_affinity_matrix", &manifold_forest<FEATT>::get_affinity_matrix_py) \
        .add_property("params",make_function(&manifold_forest<FEATT>::get_params_py, return_internal_reference<>()), \
                      &manifold_forest<FEATT>::set_params_py) \
        .add_property("stats", make_function(&manifold_forest<FEATT>::get_stats_py, return_internal_reference<>())) \
        .def("get_tree", &manifold_forest<FEATT>::get_tree, \
                return_internal_reference<>())

#define EXPOSE_REGRESSION_FORESTS(FEATT, LABELT, SPLITT, SUFFIX) \
    class_<regression_node<FEATT, LABELT, SPLITT>, bases<decision_node> >("RegressionNode" SUFFIX, no_init) \
        .add_property("dist", make_function(&regression_node<FEATT, LABELT, SPLITT>::get_sample_distribution, return_internal_reference<>())) \
        .add_property("split", make_function(&regression_node<FEATT, LABELT, SPLITT>::get_splitter, return_internal_reference<>())) \
        .def("is_leaf", &regression_node<FEATT, LABELT, SPLITT>::is_leaf) \
        .def("is_root", &regression_node<FEATT, LABELT, SPLITT>::is_root) \
        .def("max_depth", &regression_node<FEATT, LABELT, SPLITT>::max_depth) \
        .add_property("l", make_function(&regression_node<FEATT, LABELT, SPLITT>::get_left, return_internal_reference<>())) \
        .add_property("r", make_function(&regression_node<FEATT, LABELT, SPLITT>::get_right, return_internal_reference<>())); \
	class_<regression_tree<FEATT, LABELT, SPLITT>, bases<decision_tree> >("RegressionTree" SUFFIX) \
        .def("forest", &regression_tree<FEATT, LABELT, SPLITT>::get_forest, return_internal_reference<>()) \
        .add_property("root", make_function(&regression_tree<FEATT, LABELT, SPLITT>::get_root, return_internal_reference<>())) \
        .def("max_depth", &regression_tree<FEATT, LABELT, SPLITT>::max_depth); \
    class_<regression_forest<FEATT, LABELT, SPLITT>, bases<decision_forest> >("RegressionForest" SUFFIX, init<>()) \
        .def(init<regression_forest_params>()) \
        .def("_train", &regression_forest<FEATT, LABELT, SPLITT>::train_py) \
        .def("_predict", &regression_forest<FEATT, LABELT, SPLITT>::predict_py_var_num_trees_max_depth) \
        .def("_predict", &regression_forest<FEATT, LABELT, SPLITT>::predict_py_var_leaf_indices_num_trees_max_depth) \
        .add_property("params", make_function(&regression_forest<FEATT, LABELT, SPLITT>::get_params_py, return_internal_reference<>()), \
                      &regression_forest<FEATT, LABELT, SPLITT>::set_params_py) \
        .add_property("stats", make_function(&regression_forest<FEATT, LABELT, SPLITT>::get_stats_py, return_internal_reference<>())) \
        .def("get_tree", &regression_forest<FEATT, LABELT, SPLITT>::get_tree, \
                return_internal_reference<>())
    
    EXPOSE_MANIFOLD_FORESTS(double, "");
    EXPOSE_REGRESSION_FORESTS(double, double, axis_aligned_split_finder, "");
    EXPOSE_REGRESSION_FORESTS(double, double, hyperplane_split_finder, "Hyp");
    EXPOSE_REGRESSION_FORESTS(double, double, pairwise_diff_split_finder, "Prws");
    EXPOSE_REGRESSION_FORESTS(double, double, ctf_pairwise_singular_split_finder, "PrwsSingCtF");
    EXPOSE_REGRESSION_FORESTS(double, double, ctf_pairwise_only_split_finder, "PrwsCtF");

    // EXPOSE_MANIFOLD_FORESTS(float, "Flt");
    EXPOSE_REGRESSION_FORESTS(float, float, axis_aligned_split_finder, "Flt");
    EXPOSE_REGRESSION_FORESTS(float, float, single_double_spltfndr, "FltSingDoub");
    EXPOSE_REGRESSION_FORESTS(float, float, hyperplane_split_finder, "FltHyp");
    EXPOSE_REGRESSION_FORESTS(float, float, smart_hyperplane_split_finder, "FltSmrtHyp");
    EXPOSE_REGRESSION_FORESTS(float, float, pairwise_diff_split_finder, "FltPrws");
    EXPOSE_REGRESSION_FORESTS(float, float, ctf_pairwise_singular_split_finder, "FltPrwsSingCtF");
    EXPOSE_REGRESSION_FORESTS(float, float, ctf_pairwise_only_split_finder, "FltPrwsCtF");
    EXPOSE_REGRESSION_FORESTS(float, float, ncc_levels_2468_pairwise_spltfndr, "FltNcc");


		
}
