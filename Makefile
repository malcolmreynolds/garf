# User specific configuration here
EIGEN_INC_DIR = /usr/local/include/eigen3

# Must point to gtest library direction which contains the include/ subdirectory
GTEST_DIR = /Users/malc/opt/gtest  

CXX=clang++
CXXFLAGS += -O3 -g -Wall -Wextra -std=c++0x -stdlib=libc++ -ferror-limit=3 -Wno-unused-parameter 
LDFLAGS += -stdlib=libc++ -lboost_serialization-mt  -ltbb
TEST_LDFLAGS +=  -lpthread

# Eigen
CXXFLAGS += -I$(EIGEN_INC_DIR)  

# Google Test stuff - taken from the gtest sample1 makefile
CPPFLAGS += -I$(GTEST_DIR)/include
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

 

TESTS = bin/gaussian_tests bin/forest_tests

all : gtest $(TESTS)

clean :
	rm -rf objs/*.o bin/*

REG_FRST_HDRS = garf/*.hpp garf/util/*.hpp
REG_FRST_SRC = garf/*.cpp garf/*.hpp garf/splits/*.cpp garf/util/*.hpp

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



objs/forest_tests.o : tests/forest_tests.cpp $(REG_FRST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -c tests/forest_tests.cpp -o $@


objs/gaussian_tests.o : tests/gaussian_tests.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -c $^ -o $@

bin/gaussian_tests : objs/gaussian_tests.o bin/gtest.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_LDFLAGS) $(LDFLAGS) $^ -o $@

bin/forest_tests : objs/forest_tests.o bin/gtest.a $(REG_FRST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_LDFLAGS) $(LDFLAGS) objs/forest_tests.o bin/gtest.a -o $@

