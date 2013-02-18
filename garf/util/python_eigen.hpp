#ifndef GARF_UTIL_PYTHON_EIGEN_HPP
#define GARF_UTIL_PYTHON_EIGEN_HPP

// The include directory for this should be automatically taken care of
// by np.get_include() in the setup.py file
#include "numpy/arrayobject.h"

namespace garf {

    template<typename T> int eigen_type_to_np(T t) { throw std::logic_error("eigen_type_to_np not implemented for this type, "); }
    template<> int eigen_type_to_np(float f) { return NPY_FLOAT; }
    template<> int eigen_type_to_np(double d) { return NPY_FLOAT64; }
    template<> int eigen_type_to_np(eigen_idx_t i) { return NPY_LONG; }

    template<typename T>
    PyObject* eigen_to_numpy_copy(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & mtx_eig) {
        eigen_idx_t rows = mtx_eig.rows();
        eigen_idx_t cols = mtx_eig.cols();

        npy_intp dims[2];
        dims[0] = rows;
        dims[1] = cols;

        if ((rows == 0) || (cols == 0)) {
            throw std::logic_error("cannot convert a vector with zero elements");
        }

        // Use the first coeff as an argument to the above templated function
        int numpy_type = eigen_type_to_np(mtx_eig.coeff(0, 0));

        PyObject* mtx_numpy = PyArray_SimpleNew(2, dims, numpy_type);

        for (eigen_idx_t i = 0; i < rows; i++) {
            for (eigen_idx_t j = 0; j < cols; j++) {
                *((T*)PyArray_GETPTR2(mtx_numpy, i, j)) = mtx_eig.coeff(i, j);
            }
        }
        return mtx_numpy;
    }

    // Need a way to make a temporary non-modifiable Eigen object from a matrix
    // passed in by Numpy..

    template<typename T>
    const Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> * numpy_to_eigen_map(PyObject * numpy_array) {

        if (!PyArray_Check(numpy_array)) {
            throw std::invalid_argument("supplied python object is not a Numpy Array");
        }

        int num_dims = PyArray_NDIM(numpy_array);

        if (num_dims > 2) {
            throw std::invalid_argument("cannot convert an array with dimensions > 2");
        } else if (num_dims == 0) {
            throw std::invalid_argument("array has zero dimensions ?!?");
        }

        // This is safe because we know that the number of dimensions 
        eigen_idx_t rows, cols;
        npy_intp * dims = PyArray_DIMS(numpy_array);
        rows = dims[0];
        if (num_dims == 1) {
            cols = 1; // do a column vector by default
        } else {
            cols = dims[1];
        }

        // Sometimes even asking python to do a fortran order conversion doesn't seem to work
        // when we have a singleton dimension (ie a N x 1 array), but in that case contiguity doesn't matter
        if (!PyArray_ISFORTRAN(numpy_array) &&
            (rows != 1) && (cols != 1)) {
            throw std::invalid_argument("cannot perform conversion, contiguity is a problem!");
        }

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> * ret_val = 
            new Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>((T*)PyArray_DATA(numpy_array), rows, cols);

        return ret_val;
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