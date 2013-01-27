#include "gtest/gtest.h"

#include <Eigen/Dense>
#include <Eigen/Core>
// #include <Eigen/VectorwiseOp.h>


using Eigen::Matrix;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;



#include "garf/util/multi_dim_gaussian.hpp"

const double tol = 0.00001;

TEST(MDGTest, HelloWorld) {
    MatrixXd data(4, 3);
    data.setRandom();

    garf::multi_dim_gaussian<3> mdg;

    mdg.fit_params(data);

    EXPECT_NEAR(mdg.mean(0), -0.3270989, tol);
    EXPECT_NEAR(mdg.mean(1), -0.261182, tol);
    EXPECT_NEAR(mdg.mean(2), 0.258454, tol);
    // std::cout << "mean = " << mdg.mean.transpose() << std::endl;
}

GTEST_API_ int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}