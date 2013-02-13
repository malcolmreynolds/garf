CXX=clang++
# Flags passed to the C++ compiler.
CXXFLAGS += -O3 -g -Wall -Wextra -std=c++0x -stdlib=libc++ -ferror-limit=3


# Eigen
CXXFLAGS += -I/usr/local/include/eigen3  -Wno-unused-parameter -I/opt/include/


# Google Test stuff - taken from the gtest sample1 makefile
GTEST_DIR = /Users/malc/opt/gtest
CPPFLAGS += -I$(GTEST_DIR)/include
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# # Pantheios stuff
# PAN_LDFLAGS = -lpantheios.1.core.gcc42 \
#               -lpantheios.1.be.fprintf.gcc42 \
#               -lpantheios.1.bec.fprintf.gcc42 \
#               -lpantheios.1.util.gcc42 \
#               -lpantheios.1.fe.N.gcc42 \
#               -L/opt/lib

# Log4cplus
LDFLAGS += -llog4cplus

# Google logging
CPPFLAGS += -I/opt/include
LDFLAGS += -stdlib=libc++ -lboost_serialization-mt  -ltbb
TEST_LDFLAGS +=  -lpthread

 

TESTS = bin/gaussian_tests bin/forest_tests

all : gtest $(TESTS)

clean :
	rm -rf objs/*.o bin/*



REG_FRST_HDRS = garf/*.hpp garf/util/*.hpp
REG_FRST_SRC = garf/*.cpp garf/*.hpp garf/impl/*.cpp garf/util/*.hpp



# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
objs/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest-all.cc -o $@

objs/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest_main.cc -o $@

bin/gtest.a : objs/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

bin/gtest_main.a : objs/gtest-all.o objs/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

gtest : bin/gtest_main.a bin/gtest.a


# objs/regression_forest.o : garf/regression_forest.cpp garf/regression_forest.hpp
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c garf/regression_forest.cpp -o $@
# objs/regression_tree.o : garf/regression_tree.cpp garf/regression_forest.hpp
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c garf/regression_tree.cpp -o $@
# objs/regression_node.o : garf/regression_node.cpp garf/regression_forest.hpp
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c garf/regression_node.cpp -o $@



objs/forest_tests.o : tests/forest_tests.cpp $(REG_FRST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -c tests/forest_tests.cpp -o $@


objs/gaussian_tests.o : tests/gaussian_tests.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -c $^ -o $@

bin/gaussian_tests : objs/gaussian_tests.o bin/gtest.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_LDFLAGS) $(LDFLAGS) $^ -o $@

bin/forest_tests : objs/forest_tests.o bin/gtest.a $(REG_FRST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_LDFLAGS) $(LDFLAGS) objs/forest_tests.o bin/gtest.a -o $@

