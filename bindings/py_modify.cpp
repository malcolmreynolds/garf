#include <boost/python.hpp>
using namespace boost::python;

#include <pyublas/numpy.hpp>

#include <iostream>

// #define BUILD_PYTHON_BINDINGS

class A {
public:
    pyublas::numpy_matrix<double> test1(pyublas::numpy_matrix<double> blah); 
    void test2(pyublas::numpy_matrix<double> blah);
    void test3(pyublas::numpy_matrix<double>& blah);
    /*
    void test4helper(pyublas::numpy_matrix<double> blah);
    void test4(boost::numeric::ublas::matrix<double>& blah);
    */
};

pyublas::numpy_matrix<double> A::test1(pyublas::numpy_matrix<double> blah) {
    std::cout << "test1" << std::endl;
    blah(0,0) = 42;
    return blah;
}

void A::test2(pyublas::numpy_matrix<double> blah) {
    std::cout << "test2" << std::endl;
    test3(blah);
}

void A::test3(pyublas::numpy_matrix<double>& blah) {
    std::cout << "test3" << std::endl;
    blah(0,0) = 32;
}
/*
void A::test4helper(pyublas::numpy_matrix<double> blah) {
    std::cout << "test4helper" << std::endl;
    test4(reinterpret_cast<boost::numeric::ublas::matrix<double> >(blah));
}

void A::test4(boost::numeric::ublas::matrix<double>& blah) {
    std::cout << "test4" << std::endl;
    blah(0,0) = 22;
}
*/

BOOST_PYTHON_MODULE(py_modify) {
	class_<A>("A")
        .def("test1", &A::test1)
        .def("test2", &A::test2);
//        .def("test4", &A::test4helper);

}