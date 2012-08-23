#include <boost/shared_array.hpp>

#include <iostream>

class A {
public:
	boost::shared_array<double> doubles;
	A(unsigned int size = 5) {
		std::cout << "A::A()" << std::endl
   			      << "size = " << size << std::endl
    			  << "doubles.get() == " << doubles.get() << std::endl;
			doubles.reset(new double[size]);
			doubles[0] = 42.0;
		
	}
	A(const A& copy) {
		std::cout << "A::A(A& copy)" << std::endl;
		this->doubles = copy.doubles;
	}
	~A() {
		std::cout << "A::~A()" << std::endl;
	}
};

int main(int argc, char* argv[]) {
	std::cout << "about to create A" << std::endl;
	A myA;
	std::cout << "A.doubles[0] = " << myA.doubles[0] << std::endl;
	
	{
		std::cout << "creating otherA" << std::endl;
		A otherA = myA;
		otherA.doubles[0] = -1.0;
		std::cout << "otherA.doubles[0] = " << otherA.doubles[0] << std::endl;
		
	}
	std::cout << "otherA out of scope" << std::endl;
	std::cout << "A.doubles[0] = " << myA.doubles[0] << std::endl;
}