#ifndef GARF_UTIL_MULTI_DIM_GAUSSIAN_HPP
#define GARF_UTIL_MULTI_DIM_GAUSSIAN_HPP

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
using Eigen::VectorXi;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

namespace garf {

    template<int dimensions>
    class MultiDimGaussian {
    public:

        uint32_t dimensionality() const { return dimensions; }
        Matrix<double, dimensions, 1> mean;
        Matrix<double, dimensions, dimensions> cov;

        inline MultiDimGaussian() { initialise_params(); }

        inline void initialise_params() {
            mean.setZero();
            cov.setZero();
        }

        inline MultiDimGaussian(Matrix<double, dimensions, 1> _mean,
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

            // recurrence relation, see http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
            for (uint32_t k = 1; k < num_input_datapoints; k++) {
                mean_k_minus_1 = mean;
                // cov_k_minus_1 = cov;

                // update the mean
                mean = mean_k_minus_1 + (input_data.row(k).transpose() - mean_k_minus_1) / static_cast<double>(k+1);

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
            }
        }

        /* As above, but allows us to also pass a vector of indices indicating only
           certain rows of the data matrix should be considered */
        inline void fit_params(const MatrixXd & input_data, const VectorXi & valid_indices) {
            check_data_dimensionality(input_data);
            uint32_t num_input_datapoints = valid_indices.size();

            Matrix<double, dimensions, 1> mean_k_minus_1;
            initialise_params();

            mean = input_data.row(valid_indices(0));

            for (uint32_t k = 1; k < num_input_datapoints; k++) {
                int data_idx = valid_indices(k);
                mean_k_minus_1 = mean;
                // cov_k_minus_1 = cov;

                // update the mean
                mean = mean_k_minus_1 + (input_data.row(data_idx).transpose() - mean_k_minus_1) / static_cast<double>(k + 1);

                // update the covariance
                for (uint32_t d1 = 0; d1 < dimensions; d1++) {
                    double d1_diff = input_data(data_idx, d1) - mean_k_minus_1(d1);
                    for(uint32_t d2 = d1; d2 < dimensions; d2++) {
                        double addition = (k / static_cast<double>(k+1)) * d1_diff
                                        * (input_data(data_idx, d2) - mean_k_minus_1(d2));
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

        void print() const {
            std::cout << "MVN dims = " << dimensions << std::endl 
                << "mean:" << std::endl << mean <<  std::endl
                << "cov:" << std::endl  << cov << std::endl;
        }
    };

    // if we don't know the dimensionality in advance
    class MultiDimGaussianX {
    public:
        const uint32_t dimensions;
        MatrixXd mean;
        MatrixXd cov;

        inline MultiDimGaussianX(uint32_t _dimensions) : dimensions(_dimensions) {
            initialise_params();
        }

        inline void initialise_params() {
            mean = MatrixXd(dimensions, 1);
            cov = MatrixXd(dimensions, dimensions);
            mean.setZero();
            cov.setZero();
        }

        inline MultiDimGaussianX(uint32_t _dimensions,
                                MatrixXd _mean,
                                MatrixXd _cov) : dimensions(_dimensions) {
            // Check size of the provided mean and cov..
            // This complicated test allows to have the mean vector either way round
            if (!(((_mean.rows() == dimensions) && (_mean.cols() == 1)) ||
                  ((_mean.rows() == 1) && (_mean.cols() == dimensions)))) {
                throw std::invalid_argument("provided mean is wrong shape");
            }
            if ((_cov.rows() != dimensions) || (_cov.cols() != dimensions)) {
                throw std::invalid_argument("provided cov is wrong shape");
            }

            // Store the mean as a column vector            
            mean = _mean;
            mean.resize(dimensions, 1);
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
            MatrixXd mean_k_minus_1(dimensions, 1);
            initialise_params();

            mean = input_data.row(0);

            // recurrence relation, see http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
            for (uint32_t k = 1; k < num_input_datapoints; k++) {
                mean_k_minus_1 = mean;
                // cov_k_minus_1 = cov;

                // update the mean
                mean = mean_k_minus_1 + (input_data.row(k).transpose() - mean_k_minus_1) / static_cast<double>(k+1);

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

                // We have only calculated the upper triangular portion of the answer. Copy into the lower triangular bit
                for (uint32_t d1 = 0; d1 < dimensions; d1++) { 
                    for (uint32_t d2 = (d1 + 1); d2 < dimensions; d2++) {
                        cov(d2, d1) = cov(d1, d2);
                    }
                }
            }

            if (num_input_datapoints > 1) {
                cov /= (num_input_datapoints - 1);
            }
        }

        /* As above, but allows us to also pass a vector of indices indicating only
           certain rows of the data matrix should be considered */
        inline void fit_params(const MatrixXd & input_data, const VectorXi & valid_indices) {
            check_data_dimensionality(input_data);
            uint32_t num_input_datapoints = valid_indices.size();

            MatrixXd mean_k_minus_1(dimensions, 1);
            initialise_params();

            mean = input_data.row(valid_indices(0));

            for (uint32_t k = 1; k < num_input_datapoints; k++) {
                int data_idx = valid_indices(k);
                mean_k_minus_1 = mean;

                // update the mean
                mean = mean_k_minus_1 + (input_data.row(data_idx).transpose() - mean_k_minus_1) / static_cast<double>(k + 1);

                // update the covariance
                for (uint32_t d1 = 0; d1 < dimensions; d1++) {
                    double d1_diff = input_data(data_idx, d1) - mean_k_minus_1(d1);
                    for(uint32_t d2 = d1; d2 < dimensions; d2++) {
                        double addition = (k / static_cast<double>(k+1)) * d1_diff
                                        * (input_data(data_idx, d2) - mean_k_minus_1(d2));
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

        void print() const {
            std::cout << "MVN dims = " << dimensions << std::endl 
                << "mean:" << std::endl << mean <<  std::endl
                << "cov:" << std::endl  << cov << std::endl;
        }

        friend std::ostream& operator<< (std::ostream& stream, const MultiDimGaussianX& mdg) {
            stream << "[MVN:mean[" << mdg.mean.transpose() << "]:cov[";
            for (uint32_t r = 0; r < dimensions; r++) {
                stream << mdg.mean.row(r)
                if (r != (dimensions - 1)) { // if we have more rows to go..
                    stream << "; ";
                }
            }
            stream << "]]"
        }


    };

}

#endif
