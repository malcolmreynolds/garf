#ifndef GARF_UTIL_PYTHON_EIGEN_HPP
#define GARF_UTIL_PYTHON_EIGEN_HPP

// The include directory for this should be automatically taken care of
// by np.get_include() in the setup.py file
#include "numpy/arrayobject.h"

namespace garf {


    // template<

    // Given some Eigen matrix, return a new numpy object with copied data
    PyObject* eigen_to_numpy_copy(const MatrixXd & mtx_eig) {   
        eigen_idx_t rows = mtx_eig.rows();
        eigen_idx_t cols = mtx_eig.cols();

        npy_intp * dims = new npy_intp[2];

        dims[0] = rows;
        dims[1] = cols;

        PyObject* mtx_numpy = PyArray_SimpleNew(2, dims, NPY_FLOAT);

        for (eigen_idx_t i = 0; i < rows; i++) {
            for (eigen_idx_t j = 0; j < cols; j++) {
                *((float*)PyArray_GETPTR2(mtx_numpy, i, j)) = mtx_eig.coeff(i, j);
                std::cout << "mtx_eig(" << i << "," << j << ") = " << mtx_eig.coeff(i, j) << std::endl;
                std::cout << "mtx_numpy(" << i << "," << j << ") = " << *((float *)(PyArray_GETPTR2(mtx_numpy, i, j))) << std::endl;
            }
        }

        // We DONT need to incref the matrix we are returning here - the python
        // interpreter just owns the thing as soongets sorted out nicely.
        // Thanks, python. Thython.
        delete [] dims;
        std::cout << "mtx_numpy = "  << mtx_numpy << std::endl;
        return mtx_numpy;
    }

    PyObject* get_new_numpy_array() {
        npy_intp * dims = new npy_intp[2];
        dims[0] = 3;
        dims[1] = 4;
        PyObject* data = PyArray_SimpleNew(2, dims, NPY_FLOAT);

        delete [] dims;

        return data;
    }
}

#endif