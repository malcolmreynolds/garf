#ifndef GARF_MULTIVARIATE_NORMAL_HPP
#define GARF_MULTIVARIATE_NORMAL_HPP

#include <stdint.h>
#include <stdexcept>
#include <cmath>

#include "types.hpp"
#include "determinant.hpp"
#include "forest_utils.hpp"
#include "utils.hpp"

#ifdef USING_OPENMP
#include <omp.h>
#endif

#include <boost/numeric/ublas/matrix_proxy.hpp>

//#define WITH_ITERATORS

namespace garf {
    
    using namespace boost::numeric::ublas;
    
    template<typename FeatT>
    class multivariate_normal {
        uint32_t dimensions; // dimensionality of normal
    public:
        // the k_minus_1 variables are needed for correct & stable variance calculation, see
        // http://www.johndcook.com/standard_deviation.html for more info
        typename garf_types<FeatT>::vector mu; // mean vector
        typename garf_types<FeatT>::matrix sigma; // covariance
        FeatT determinant;
        
        // constructor just takes the dimensionality
        inline multivariate_normal(uint32_t _d) :
            dimensions(_d),
            mu(_d),
            sigma(_d,_d) {
            clear();
        };

//        inline ~multivariate_normal() { }; 
        inline ~multivariate_normal() {
            // FIXME: according to the code below, we end up seeing some weird ass shit with openmp
            std::cout << "";
#ifdef VERBOSE            
#ifdef USING_OPENMP
            std::cout << (threadsafe_ostream() << "~multivariate_normal thread " 
                            << omp_get_thread_num() << " : " 
                            << this << '\n').toString();
#else
            std::cout << "~multivariate_normal : " << this << std::endl;
#endif        
#endif            
        };
        
        void fit_params(const typename garf_types<FeatT>::matrix& data, const garf_types<training_set_idx_t>::vector& valid_indices, uint32_t num_data_points);
        void fit_mean_to_all(const typename garf_types<FeatT>::matrix& data);
        void fit_mean_and_variances_to_all(const typename garf_types<FeatT>::matrix& data);
        // void fit_covariance(const typename garf_types<FeatT>::matrix& data, const garf_types<training_set_idx_t>::vector& valid_indices);
        
        inline void clear() {
            mu.clear();
            sigma.clear();
            determinant = 0.0;
        };
        
        // Unsupervised information gain, if we have no labels
        static inf_gain_t unsup_inf_gain(const multivariate_normal<FeatT>& parent_dist,
                                                 const multivariate_normal<FeatT>& left_dist,
                                                 const multivariate_normal<FeatT>& right_dist, 
                                                 uint32_t num_in_parent, uint32_t num_in_left, uint32_t num_in_right);
                                                 
#ifdef BUILD_PYTHON_BINDINGS
    inline uint32_t get_dimensions() { return dimensions; };
#endif
    };

    // fit the mean to all the data provided - when we don't need the flexibility of a valid_indices styl solution, this is fine
    template<typename FeatT>
    void multivariate_normal<FeatT>::fit_mean_to_all(const typename garf_types<FeatT>::matrix& data) {
        if (data.size2() != dimensions) {
            throw std::invalid_argument("data passed into multivariate_normal::fit_mean is of wrong dimensionality");
        }
        
        training_set_idx_t num_data_points = data.size1();

        // declare temporaries
        typename garf_types<FeatT>::vector mu_k_minus_1(dimensions);

        // New method, taken from http://www.johndcook.com/standard_deviation.html - numerical accuracy is superior
        mu.clear();
        mu_k_minus_1.clear();

        // get the first row of data
        matrix_row<const typename garf_types<FeatT>::matrix> first_data_row(data, 0);

        typename garf_types<FeatT>::vector::iterator mu_it = mu.begin();
        typename garf_types<FeatT>::vector::iterator mu_k_m_1_it;
        for (uint32_t d=0; d < dimensions; d++) {
            *mu_it++ = first_data_row(d);
        }

        // Now do the recurrence relation
        for (uint32_t k=1; k < num_data_points; k++) {
            matrix_row<const typename garf_types<FeatT>::matrix> current_data_row(data, k);
            // std::cout << "k=" << k << " data_idx = " << data_idx << std::endl;
            // The values we've already calculated are now the "k minus 1"'th
            // values, so copy the values over
            mu_k_minus_1.assign(mu);

            // update the mean
            mu_it = mu.begin();
            mu_k_m_1_it = mu_k_minus_1.begin();
            for (uint32_t d=0; d < dimensions; d++) {                
                *mu_it++ = mu_k_minus_1(d) + (data(k, d) - *mu_k_m_1_it++) / (double)(k+1);
            }
        }
    }    
    
    // Compute only the mean and variances - this is when we want to analyse the distribution of outputs over trees,
    // and we are not interested in the full covariance matrix, only the diagonal
    template<typename FeatT>
    void multivariate_normal<FeatT>::fit_mean_and_variances_to_all(const typename garf_types<FeatT>::matrix& data) {
        if (data.size2() != dimensions) {
            throw std::invalid_argument("data passed into multivariate_normal::fit_mean is of wrong dimensionality");
        }
        training_set_idx_t num_data_points = data.size1();

        // declare temporaries
        typename garf_types<FeatT>::vector mu_k_minus_1(dimensions);
        typename garf_types<FeatT>::matrix sigma_k_minus_1(dimensions, dimensions);

        // New method, taken from http://www.johndcook.com/standard_deviation.html - numerical accuracy is superior
        clear();
        mu_k_minus_1.clear();
        sigma_k_minus_1.clear();
        
        // get the first row of data
        matrix_row<const typename garf_types<FeatT>::matrix> first_data_row(data, 0);

        typename garf_types<FeatT>::vector::iterator mu_it = mu.begin();
        typename garf_types<FeatT>::vector::iterator mu_k_m_1_it;
        for (uint32_t d=0; d < dimensions; d++) {
            *mu_it++ = first_data_row(d);
        }

        // Now do the recurrence relation
        for (uint32_t data_idx=1; data_idx < num_data_points; data_idx++) {
            matrix_row<const typename garf_types<FeatT>::matrix> current_data_row(data, data_idx);
            // std::cout << "k=" << k << " data_idx = " << data_idx << std::endl;

            mu_k_minus_1.assign(mu);
            sigma_k_minus_1.assign(sigma);

            // std::cout << "mu_k_minus_1 = " << mu_k_minus_1 << "sigma_k_minus_1 = " << sigma_k_minus_1 << std::endl;

            // update the mean and the variance - note we don't need the second loop for full covariance
            mu_it = mu.begin();
            mu_k_m_1_it = mu_k_minus_1.begin();
            FeatT temp;
            for (uint32_t d=0; d < dimensions; d++) {                
                *mu_it++ = mu_k_minus_1(d) + (data(data_idx, d) - *mu_k_m_1_it++) / (double)(data_idx+1);
                // In the full covariance loop this is not needed as the temp*temp product comes from different
                // dimension indices.. however here we are just doing the diagonal, so not needed.
                temp = (data(data_idx,d) - mu_k_minus_1(d));
                sigma(d, d) = sigma_k_minus_1(d, d) + (data_idx / (double)(data_idx+1)) * temp * temp;
            }
            //std::cout << "mu_k = " << mu << " sigma = " << sigma << std::endl << std::endl;
        }

        // We can now divide sigma by k-1 to get the variance
        if (num_data_points > 1) {  
            sigma /= (num_data_points-1);
        }        
        // don't bother with determinant in this case
    }


    
    // fit both the parameters to the data. num_data_points specifies how many of the indices are valid - this allows
    // us to preallocate the valid_indices with some maximum length, but we might then want to calculate the num
    template<typename FeatT>
    void multivariate_normal<FeatT>::fit_params(const typename garf_types<FeatT>::matrix& data,
                                                      const garf_types<training_set_idx_t>::vector& valid_indices,
                                                      uint32_t num_data_points) {
        if (data.size2() != dimensions) {
            throw std::invalid_argument("data passed into multivariate_normal::fit_mean is of wrong dimensionality");
        }

        // declare temporaries
        typename garf_types<FeatT>::vector mu_k_minus_1(dimensions);
        typename garf_types<FeatT>::matrix sigma_k_minus_1(dimensions, dimensions);
        
        // New method, taken from http://www.johndcook.com/standard_deviation.html - numerical accuracy is superior
        mu.clear();
        mu_k_minus_1.clear();
        sigma.clear();
        sigma_k_minus_1.clear();
        
        // Set mu_k to the first data point.. the initialisation for sigma_k is zeros
        garf_types<training_set_idx_t>::vector::const_iterator valid_indices_it = valid_indices.begin();
        training_set_idx_t first_data_point_index = *valid_indices_it++;
        
        // get the first row of data
        matrix_row<const typename garf_types<FeatT>::matrix> first_data_row(data, first_data_point_index);

        
        typename garf_types<FeatT>::vector::iterator mu_it = mu.begin();
        typename garf_types<FeatT>::vector::iterator mu_k_m_1_it;
        for (uint32_t d=0; d < dimensions; d++) {
            *mu_it++ = first_data_row(d);
        }

        // Now do the recurrence relation
        for (uint32_t k=1; k < num_data_points; k++) {
            training_set_idx_t data_idx = *valid_indices_it++;
#ifdef WITH_ITERATORS
            matrix_row<const typename garf_types<FeatT>::matrix> current_data_row(data, data_idx);
#endif
            // std::cout << "k=" << k << " data_idx = " << data_idx << std::endl;
            // The values we've already calculated are now the "k minus 1"'th
            // values, so copy the values over
            mu_k_minus_1.assign(mu);
            sigma_k_minus_1.assign(sigma);
            
            // std::cout << "mu_k_minus_1 = " << mu_k_minus_1 << "sigma_k_minus_1 = " << sigma_k_minus_1 << std::endl;
            
            // update the mean
            mu_it = mu.begin();
            mu_k_m_1_it = mu_k_minus_1.begin();
            for (uint32_t d=0; d < dimensions; d++) {                
#ifdef WITH_ITERATORS                
                *mu_it++ = mu_k_minus_1(d) + (current_data_row(d) - *mu_k_m_1_it++) / (double)(k+1);
#else       
                *mu_it++ = mu_k_minus_1(d) + (data(data_idx, d) - *mu_k_m_1_it++) / (double)(k+1);
#endif
            }
            
            // update the variance - see http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
            for (uint32_t d1=0; d1 < dimensions; d1++) {
                for (uint32_t d2=0; d2 < dimensions; d2++) {
#ifdef WITH_ITERATORS
                    sigma(d1, d2) = sigma_k_minus_1(d1, d2) + (k / (double)(k+1)) * (current_data_row(d1) - mu_k_minus_1(d1))
                                                                                  * (current_data_row(d2) - mu_k_minus_1(d2));
#else   
                    sigma(d1, d2) = sigma_k_minus_1(d1, d2) + (k / (double)(k+1)) * (data(data_idx, d1) - mu_k_minus_1(d1))
                                                                                  * (data(data_idx, d2) - mu_k_minus_1(d2));
            
#endif                                                                                  
                    // std::cout << "set sigma("<<d1<<","<<d2<<") = "<< sigma(d1,d2)<< std::endl;
//                    std::cout << "  = "<< (k/ (k+1)
                }
            }
            //std::cout << "mu_k = " << mu << " sigma = " << sigma << std::endl << std::endl;
        }
        
        // We can now divide sigma by k-1 to get the variance
        if (num_data_points > 1) {  
            sigma /= (num_data_points-1);
        }
        // std::cout << "final sigma = " << sigma << std::endl;
        determinant = util::lu_det(sigma);
#ifdef VERBOSE
        if (determinant <= 0 && (num_data_points > 1)) {
            // This should never happen mathematically.. but it just does sometimes. Seems to often be when we
            // have a very small amount of data, determinant should be very small but actually ends up a small
            // negative number or zero. This really sucks, but I'm already using the 'most numerically stable'
            // method of computing the covariance matrix, so at this point I have to admit defeat and write workarounds
            // for this case (see the information gain function - extra important there because zero determinant
            // leads to infinite information gain!). I've seen the same data do exactly the same in matlab, so
            // this is *NOT* a bug in my code.
            std::cout << "determinant computed as " << determinant << std::endl
                << "sigma = " << sigma << std::endl
                << "mu = " << mu << std::endl
                << "num data points = " << num_data_points << std::endl
                << "data = "<< std::endl;
            print_matrix_selected_indices_limit<FeatT>(std::cout, data, valid_indices, num_data_points);
        
//            throw std::logic_error("determinant is zero!");
        }
#endif 
    }

    // template<typename FeatT>
    // inf_gain_t multivariate_normal<FeatT>::sup_reg_inf_gain(const multivariate_normal<FeatT>& parent_dist,
    //                                                         const multivariate_normal<FeatT>& left_dist)

    // See Criminisi decision forest paper, page 71
    template<typename FeatT>
    inf_gain_t multivariate_normal<FeatT>::unsup_inf_gain(const multivariate_normal<FeatT>& parent_dist,
                                                                  const multivariate_normal<FeatT>& left_dist,
                                                                  const multivariate_normal<FeatT>& right_dist, 
                                                                  uint32_t num_in_parent, uint32_t num_in_left, uint32_t num_in_right) {

        // std::cout << "doing information gain, parent_dist.sigma = " << parent_dist.sigma << std::endl
        //     << "left_dist.sigma = " << left_dist.sigma << std::endl
        //     << "right_dist.sigma = " << right_dist.sigma << std::endl;

        // if the split only take out one or other of the data set then it's quite whack - return -Inf gain
        if ((num_in_left == 1) || (num_in_right == 1)) {
            double i_gain = -std::numeric_limits<double>::infinity();
            // std::cout << "only one node in left or right subtree, so gain = " << i_gain << std::endl;
            return i_gain;
        }

        //compute the determinants of all the covariance matrices
        double det_parent = parent_dist.determinant;
        double det_left = left_dist.determinant;
        double det_right = right_dist.determinant;
        
#ifdef VERBOSE        
        std::cout << "det_parent = " << det_parent << " parent.sigma = " << parent_dist.sigma << std::endl;
        std::cout << "det_left = " << det_left << " left.sigma = " << left_dist.sigma << std::endl;
        std::cout << "det_right = " << det_right << " right.sigma = " << right_dist.sigma << std::endl;
#endif

        double log_det_parent = log(det_parent);
        double log_det_left = log(det_left);
        double log_det_right = log(det_right);
        
#ifdef VERBOSE  
        std::cout << "log_det_parent = " << log_det_parent << std::endl;
        std::cout << "log_det_left = " << log_det_left << std::endl;
        std::cout << "log_det_right = " << log_det_right << std::endl; 
#endif
        //criminis page 71, figure 5.2
        double i_gain = log_det_parent;
        i_gain -= (num_in_left * log_det_left) / (double)num_in_parent;
        i_gain -= (num_in_right * log_det_right) / (double)num_in_parent;
        
        if (i_gain == std::numeric_limits<inf_gain_t>::infinity()) {
            // If this happens then one of the individual left/right distributions has ended up with negative infinity
            // entropy. See the comments inside fit_params above for details about why this happens
#ifdef VERBOSE
            std::cout << "det_parent = " << det_parent << " parent.sigma = " << parent_dist.sigma << std::endl;
            std::cout << "det_left = " << det_left << " left.sigma = " << left_dist.sigma << std::endl;
            std::cout << "det_right = " << det_right << " right.sigma = " << right_dist.sigma << std::endl;
            std::cout << "log_det_parent = " << log_det_parent << std::endl;
            std::cout << "log_det_left = " << log_det_left << std::endl;
            std::cout << "log_det_right = " << log_det_right << std::endl;
            
            std::cout << "i_gain = " << i_gain << std::endl;
            std::cout << "returning minus infinity" << std::endl;
#endif            
            return -std::numeric_limits<inf_gain_t>::infinity();
        }
        
        return i_gain;
    }
}




#endif 