#include <stdexcept>

// gets us the slow pyublas vectors for timing
//#define BUILD_PYTHON_BINDINGS

// #include "multivariate_normal.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "../forest_utils.hpp"
#include "../training_set.hpp"
#include "../manifold_forest.hpp"



using namespace garf;
using namespace boost::numeric::ublas;
using namespace boost::random;

int main(int argc, char** argv) {

    try {

    	//5 samples with dimensionality 3
        uint32_t num_samples = 200;
        uint32_t dims = 5;
        uint32_t label_dims = 2;
	
        boost::shared_ptr<matrix<double> > m_p(new matrix<double>(num_samples,dims));
	    {
            matrix<double>& m = *m_p;
	
        	//first feature
            m(0,0) = 3;
            m(1,0) = 1;
            m(2,0) = 3.2;
            m(3,0) = 0.8;
            m(4,0) = 2.87;
    
            //second feature
            m(0,1) = 2.5;
            m(1,1) = 0.4;
            m(2,1) = 3.1;
            m(3,1) = 0.8;
            m(4,1) = 3.0;
    
    
    
            //third, useless feature
            mt19937 gen;
            normal_distribution<double> nd;
        //    variate_generator<mt19937&, normal_distribution<> > var_nor(gen, nd);
            for (uint32_t i=0; i < dims; i++) {
                for (uint32_t j=0; j < num_samples; j++) {
                    m(j,i) = nd(gen);
                }
            }
        }
        std::cout << "training on " << *m_p << std::endl;
        
        boost::shared_ptr<matrix<double> > lbls_p(new matrix<double>(num_samples, label_dims));
        {
            matrix<double>& lbls = *lbls_p;
            
            mt19937 gen;
            normal_distribution<double> nd;
            
            for (uint32_t i=0; i < num_samples; i++) {
                for (uint32_t j=0; j < label_dims; j++) {
                    lbls(i,j) = nd(gen);
                }
            }
        }
    
    
	    

    	boost::shared_ptr<manifold_forest<double> > mf(new manifold_forest<double>());
	    // the training set gets stored in a shared_ptr inside the forest as well, so keep like this.
        //boost::shared_ptr<training_set<double> > t_set(reinterpret_cast<training_set<double>* >(new unsupervised_training_set<double>(m_p)));
        boost::shared_ptr<training_set<double> > t_set(new unsupervised_training_set<double>(m_p));
        // boost::shared_ptr<training_set<double> > t_set(new multi_supervised_regression_training_set<double,double>(m_p, lbls_p));
        
        std::cout << "about to do training" << std::endl;
	    mf->train(t_set);

        t_set.reset(); // get rid of the reference to training set - it should
        
        mf->compute_affinity_matrix();
        const matrix<double> affinity = *(mf->_affinity_matrix);
        std::cout << "affinity computed as " << affinity << std::endl;
        
    }
    catch(std::exception e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }


	
}