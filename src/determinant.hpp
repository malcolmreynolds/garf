#ifndef GARF_DETERMINANT_HPP
#define GARF_DETERMINANT_HPP

//taken from http://lists.boost.org/MailArchives/ublas/2005/12/0916.php
//FIXME: check the licensing for htis!!

/*
#if BOOST_VERSION < 103301 // missing includes in lu.hpp 
#include "boost/numeric/ublas/vector.hpp" 
#include "boost/numeric/ublas/vector_proxy.hpp" 
#include "boost/numeric/ublas/matrix.hpp" 
#include "boost/numeric/ublas/matrix_proxy.hpp" 
#include "boost/numeric/ublas/triangular.hpp" 
#include "boost/numeric/ublas/operation.hpp" 
#endif 
*/
#include <boost/numeric/ublas/lu.hpp> 
// forget about JFR_PRECOND macros... 

namespace garf { namespace util {

    /** General matrix determinant. 
     * It uses lu_factorize in uBLAS. 
     */ 
    template<class M> double lu_det(M const& m) { 
        //std::cout << "doing lu_det on " << m << std::endl;
         
        if (m.size1() != m.size2()) {
            throw std::invalid_argument("ublasExtra::lu_det: matrix must be square");
        }

        // create a working copy of the input  - seems like this has to be boost::numeric::ublas
        // types, if we do pyublas types we get memory corruption and general bad times.
        boost::numeric::ublas::matrix<double> mLu(m.size1(), m.size2()); 
        mLu.assign(m);

        boost::numeric::ublas::permutation_matrix<std::size_t> pivots(m.size1());         
        // std::cout << "before factorize" << std::endl 
        //     << "m   = " << m << std::endl
        //     << "mLu = " << mLu << std::endl
        //     << "pivots = " << pivots << std::endl;
        
        boost::numeric::ublas::lu_factorize(mLu, pivots); 
        // std::cout << "done factorize" <<std::endl
        //     << "m   = " << m << std::endl
        //     << "mLu = " << mLu << std::endl
        //     << "pivots = " << pivots << std::endl; 
        
        double det = 1.0; 
        for (std::size_t i=0; i < pivots.size(); ++i) { 
            if (pivots(i) != i) {
                det *= -1.0; 
            }
            det *= mLu(i,i); 
        } 
        // std::cout << "returning determinant " << det << std::endl;
        return det; 
      }
      
} }

#endif