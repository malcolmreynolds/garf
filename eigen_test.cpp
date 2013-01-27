#include <iostream>
#include <stdexcept>
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
using Eigen::Matrix3d;
using Eigen::MatrixXd;


template<int dimensions>
class multi_dim_gaussian {
public:

    uint32_t dimensionality() const { return dimensions; }
    Matrix<double, dimensions, 1> mean;
    Matrix<double, dimensions, dimensions> cov;

    inline multi_dim_gaussian() { initialise_params(); }

    /* Initialise to isotropic Gaussian, std dev 1 */
    inline void initialise_params() {
        mean.setZero();
        cov.setZero();
    }
    inline multi_dim_gaussian(Matrix<double, dimensions, 1> _mean,
                              Matrix<double, dimensions, dimensions> _cov) {
        mean = _mean;
        cov = _cov;
    }

    inline void check_data_dimensionality(const MatrixXd & input_data) const {
        uint32_t input_data_dimensionality = input_data.cols();
        if (input_data_dimensionality != dimensions) {
            throw std::invalid_argument("input data dimensionality doesn't match in fit_params");   
        }
    }

    /**
        Fit mean and covariance to some dataset. Note this has to be a dynamic
        matrix (MatriXd) because even though we know what one dimension of the matrix
        should be (same as our dimensionality) we don't know the other one, and we
        can't template the entire program on every possible dataset size
     */
    inline void fit_params(const MatrixXd & input_data) {
        check_data_dimensionality(input_data);
        uint32_t num_input_datapoints = input_data.rows();

        // temporary for the efficient / stable computation:
        // http://www.johndcook.com/standard_deviation.html 
        Matrix<double, dimensions, 1> mean_k_minus_1;
        initialise_params();

        mean = input_data.row(0);
        std::cout << "k = 0 mean = " << mean_k_minus_1.transpose() << std::endl;
        // cov_k_minus_1.setZero();

        // recurrence relation, see http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
        for (uint32_t k = 1; k < num_input_datapoints; k++) {
            mean_k_minus_1 = mean;
            // cov_k_minus_1 = cov;

            // update the mean
            mean = mean_k_minus_1 + (input_data.row(k).transpose() - mean_k_minus_1) / static_cast<double>(k+1);
            std::cout << "k = " << k << " mean = " << mean.transpose() << std::endl;

            // update the covariance
            for (uint32_t d1 = 0; d1 < dimensions; d1++) {
                double d1_diff = input_data(k, d1) - mean_k_minus_1(d1);
                for(uint32_t d2 = d1; d2 < dimensions; d2++) {
                    double addition = (k / static_cast<double>(k+1)) * d1_diff
                                    * (input_data(k, d2) - mean_k_minus_1(d2));
                    cov(d1, d2) += addition;
                    cov(d2, d1) = cov(d1, d2); // could make this more efficient by only doing it at the end
                }
            }

            // We have only calcualted the upper triangular portion of the answer. Copy into the lower triangular bit
            for (uint32_t d1 = 0; d1 < dimensions; d1++) { 
                for (uint32_t d2 = (d1 + 1); d2 < dimensions; d2++) {
                    cov(d2, d1) = cov(d1, d2);
                }
            }
        }

        if (num_input_datapoints > 1) {
            cov /= (num_input_datapoints - 1);
        } else {
            // If we only have one, then the mean is just the only variable and the covariance is all zeros
            mean = input_data.row(0);
        }
    }

    inline void fit_params_inaccurate(const MatrixXd & input_data) {
        check_data_dimensionality(input_data);
        uint32_t num_input_datapoints = input_data.rows();

        initialise_params();

        // Two pass algorithm - calculate mean first. Use eigen to sum down
        // each column
        for (uint32_t d = 0; d < dimensions; d++) {
            mean(d) = input_data.col(d).sum();
        }
        mean /= num_input_datapoints;


        for (uint32_t i = 0; i < num_input_datapoints; i++) {
            const RowVectorXd data_point = input_data.row(i);
            for (uint32_t d1 = 0; d1 < dimensions; d1++) {
                for (uint32_t d2 = 0; d2 < dimensions; d2++) {
                    cov(d1, d2) += (data_point(d1) - mean(d1)) * (data_point(d2) - mean(d2));
                }
            }
        }

        // Unbiased estimator, because we don't know the population mean, only the sample
        // mean. Maybe I should have paid more attention in stats lessons,
        cov /= (num_input_datapoints - 1);
    }

    inline void fit_params(const MatrixXd & input_data, const VectorXd & valid_indices) {
        check_data_dimensionality(input_data);
        uint32_t num_input_datapoints = valid_indices.size();

        initialise_params();

        for (uint32_t i = 0; i < num_input_datapoints; i++) {
            mean += input_data.row(valid_indices(i));
        }

        mean /= num_input_datapoints;
    }


    void print() const {
        std::cout << "MVN dims = " << dimensions << std::endl 
            << "mean:" << std::endl << mean <<  std::endl
            << "cov:" << std::endl  << cov << std::endl;
    }

};


int main() {
    MatrixXd m(2,2);
    m(0, 0) = 0;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1,1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;

    multi_dim_gaussian<3> mdg;
    mdg.print();

    {
        Vector3d mu1(2.0, 3.0, 8.0);
        Vector3d mu2(8.0, 9.0, 12.0);
        mdg.mean = mu1;
        mdg.print();
        mdg.mean = mu2;    
        mdg.print();
    }

    mdg.print();

    MatrixXd data(4, 3);
    data.setRandom();
    std::cout << "data: " << std::endl << data << std::endl;

    std::cout << std::endl << "with accurate:"  << std::endl;
    mdg.fit_params(data);
    mdg.print();

    std::cout << std::endl << "with inaccurate:" << std::endl;
    mdg.fit_params_inaccurate(data);
    mdg.print();


    // std::cout << "constructed multi dim gaussian dims " << mdg.dimensionality() << std::endl;


}