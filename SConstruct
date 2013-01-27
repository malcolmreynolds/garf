env = Environment()

env.Replace(CXX='clang++')
env.Append(CPPPATH=['/usr/local/include/eigen3', '.'])
env.Append(CLIBPATH=['/usr/lib/gtest'])

# env.Program(target='eigen_test', source=['eigen_test.cpp'])
env.Program(target='bin/gauss_tests',
            source=['tests/multi_dim_gaussian_tests.cpp'],
            LIBS=['gtest', 'gtest_main', 'pthread'])






