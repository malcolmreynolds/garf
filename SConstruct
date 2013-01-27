import os

env = Environment()

env.Replace(CXX='clang++')
env.Append(CPPPATH=['/usr/local/include/eigen3', '.'])

# Add stuff for google Testing library. Should only need to change the top line
# to point this to wherever the library is installed
gtest_dir = '/Users/malc/opt/gtest/trunk'
env.Append(CPPPATH=[gtest_dir, gtest_dir + '/include'])

# Find all the C++ source files of the google test framework
gtest_src_files = [os.path.join(gtest_dir, 'src', f) for f in
                   os.listdir(gtest_dir + '/src') if f.endswith('.cc')]
# print "gtest_src_files:", gtest_src_files


# The following assumes that $BUILD_DIR refers to the root of the
# directory for your current build mode, e.g. "#/mybuilddir/mybuildmode"

# Build gtest library; as it is outside of our source root, we need to
# tell SCons that the directory it will refer to as
# e.g. $BUILD_DIR/gtest is actually on disk in original form as
# ../../gtest (relative to your project root directory).  Recall that
# SCons by default copies all source files into the build directory
# before building.
gtest_dir = env.Dir('$BUILD_DIR/gtest')

# Modify this part to point to gtest relative to the current
# SConscript or SConstruct file's directory.  The ../.. path would
# be different per project, to locate the base directory for gtest.
gtest_dir.addRepository(env.Dir('/Users/malc/opt/gtest/trunk'))

# Tell the gtest SCons file where to copy executables.
env['EXE_OUTPUT'] = '$BUILD_DIR'  # example, optional

# Call the gtest SConscript to build gtest.lib and unit tests.  The
# location of the library should end up as
# '$BUILD_DIR/gtest/scons/gtest.lib'
env.SConscript(env.File('scons/SConscript', gtest_dir))



# env.Program(target='eigen_test', source=['eigen_test.cpp'])
env.Program(target='bin/gauss_tests',
            source=['tests/multi_dim_gaussian_tests.cpp'] + gtest_src_files)  # ,
            # LIBS=['gtest', 'gtest_main', 'pthread'])






