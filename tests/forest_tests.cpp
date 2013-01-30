#include "gtest/gtest.h"
#include <glog/logging.h>

#include <Eigen/Dense>
#include <Eigen/Core>

using Eigen::Matrix;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

#include "garf/regression_forest.hpp"

const double tol = 0.00001;

TEST(ForestTest, RegTest1) {
    uint32_t data_elements = 10;
    uint32_t data_dims = 1;
    double noise_variance = 0.5;

    MatrixXd data(data_elements, data_dims);
    data.col(0).setLinSpaced(data_elements, -5, 5);
    LOG(INFO) << "Data initialised: " << data.transpose() << std::endl;

    MatrixXd noise(data_elements, data_dims);
    noise.setRandom();
    data.col(0) += noise_variance * noise;

    MatrixXd labels = data.cwiseProduct(data);
    LOG(INFO) << "labels initialised: " << labels.transpose() << std::endl;

    garf::RegressionForest<garf::AxisAlignedSplt, garf::AxisAlignedSplFitter> forest;
    forest.train(data, labels);

    EXPECT_TRUE(1 == 1);
}

GTEST_API_ int main(int argc, char **argv) {
    // Print everything, including INFO and WARNING
    FLAGS_stderrthreshold = 0;
    google::InitGoogleLogging(argv[0]);
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}