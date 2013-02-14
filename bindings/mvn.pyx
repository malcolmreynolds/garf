cdef extern from "garf/util/multi_dim_gaussian.hpp" namespace "garf":
    cdef cppclass MultiDimGaussianX[T]:
        initialise_params()

print("Hello World")
