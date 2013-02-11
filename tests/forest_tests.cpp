#include "gtest/gtest.h"
// #include <glog/logging.h>


#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Core>

using Eigen::Matrix;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

#define GARF_SERIALIZE_ENABLE

#include "garf/regression_forest.hpp"

const double tol = 0.00001;

void make_1d_labels_from_2d_data_squared_diff(const MatrixXd & features, MatrixXd & labels) {
    labels.col(0) = features.col(0).cwiseProduct(features.col(0)) - 
                    0.5 * features.col(1).cwiseProduct(features.col(1));
}

void make_1d_labels_from_1d_data_abs(const MatrixXd & features, MatrixXd & labels) {
    labels.col(0) = features.col(0).cwiseAbs();
}


// Given a forest with training parameters already set, clear the forest,
// generate a bunch of random training data, generate labels from this
// data with the provided function pointer and other params, then check that the forest
// 
void test_forest_with_data(garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> & forest,
                           void (* label_generator)(const MatrixXd &, MatrixXd &),
                           uint64_t num_train_datapoints, uint64_t num_test_datapoints,
                           uint64_t data_dims, uint64_t label_dims,
                           double data_scaler, double noise_variance,
                           double answer_tolerance) {
    forest.clear();

    // Generate the (uniformly distributed data)
    MatrixXd train_data(num_train_datapoints, data_dims);
    train_data.setRandom();
    train_data *= data_scaler;

    // Generate labels, add noise
    MatrixXd train_labels(num_train_datapoints, label_dims);
    label_generator(train_data, train_labels);
    MatrixXd noise(num_train_datapoints, label_dims);
    noise.setRandom();
    train_labels += noise_variance * noise;

    // Train forest
    forest.clear();
    forest.train(train_data, train_labels);

    // Make test data
    MatrixXd test_data(num_test_datapoints, data_dims);
    test_data.setRandom();
    test_data *= data_scaler;

    // Test labels
    MatrixXd test_labels(num_test_datapoints, label_dims);
    label_generator(test_data, test_labels);

    MatrixXd predicted_labels(num_test_datapoints, label_dims);
    forest.predict(test_data, &predicted_labels);

    for (uint64_t i = 0; i < num_test_datapoints; i++) {
        for (uint64_t j = 0; j < label_dims; j++) {
            EXPECT_NEAR(predicted_labels(i, j), test_labels(i, j), answer_tolerance);
        }
    }
}

// TEST(ForestTest, RegTest1) {
//     garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> forest;
//     forest.forest_options.max_num_trees = 10;
//     forest.tree_options.max_depth = 6;
//     forest.tree_options.min_sample_count = 2;

//     uint64_t num_train_datapoints = 1000;
//     uint64_t num_test_datapoints = 100;
//     uint64_t data_dims = 2;
//     uint64_t label_dims = 1;
//     double data_scaler = 2.0;
//     double noise_variance = 0.1;
//     double answer_tolerance= 1.0;

//     // 1D data, 1D labels
//     test_forest_with_data(forest, make_1d_labels_from_2d_data_squared_diff,
//                           num_train_datapoints, num_test_datapoints,
//                           data_dims, label_dims, data_scaler,
//                           noise_variance, answer_tolerance);
// }

// TEST(ForestTest, RegTest2) {
//     garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> forest;
//     forest.forest_options.max_num_trees = 10;
//     forest.tree_options.max_depth = 6;
//     forest.tree_options.min_sample_count = 2;

//     uint64_t num_train_datapoints = 1000;
//     uint64_t num_test_datapoints = 100;
//     uint64_t data_dims = 1;
//     uint64_t label_dims = 1;
//     double data_scaler = 2.0;
//     double noise_variance = 0.1;
//     double answer_tolerance = 0.1;

//     // 1D data, 1D labels
//     test_forest_with_data(forest, make_1d_labels_from_1d_data_abs,
//                           num_train_datapoints, num_test_datapoints,
//                           data_dims, label_dims, data_scaler,
//                           noise_variance, answer_tolerance);
// }


void save_and_restore_forest(const garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> & forest_to_save,
                             garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> * forest_to_load_into,
                             std::string filename) {
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << forest_to_save;
    }
    std::cout << "forest saved" << std::endl;

    std::cout << "before loading forest_to_load_into = " << *forest_to_load_into << std::endl;

    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> *forest_to_load_into;
    }
    std::cout << "forest loaded: " << *forest_to_load_into << std::endl;
}


TEST(ForestTest, Serialize) {
    uint64_t data_elements = 10;
    uint64_t data_dims = 2;
    uint64_t label_dims = 1;
    MatrixXd data(data_elements, data_dims);
    data.setRandom();

    MatrixXd labels(data_elements, label_dims);
    make_1d_labels_from_2d_data_squared_diff(data, labels);
    labels.setRandom();

    garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> forest1;
    garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> forest2;


    forest1.train(data, labels);

    // Serialises forest1 out to disk, loads it back into forest 2
    save_and_restore_forest(forest1, &forest2, "test_serialize.forest");

    EXPECT_TRUE(true);
}

GTEST_API_ int main(int argc, char **argv) {
    // Print everything, including INFO and WARNING
    // FLAGS_stderrthreshold = 0;
    // google::InitGoogleLogging(argv[0]);
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}