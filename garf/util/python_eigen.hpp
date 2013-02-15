#ifndef GARF_UTIL_PYTHON_EIGEN_HPP
#define GARF_UTIL_PYTHON_EIGEN_HPP

// The include directory for this should be automatically taken care of
// by np.get_include() in the setup.py file
#include "numpy/arrayobject.h"

namespace garf {

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