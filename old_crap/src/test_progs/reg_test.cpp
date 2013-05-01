#include <stdexcept>


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "../forest_utils.hpp"
#include "../training_set.hpp"
#include "../regression_forest.hpp"


#define VERBOSE


using namespace garf;
//using namespace boost::numeric::ublas;
using namespace boost::random;

typedef double feature_type;
typedef double label_type;

int main(int argc, char** argv) {

    try {

    	//5 samples with dimensionality 3
        const uint32_t num_training_samples = 20;
        const uint32_t num_predict_samples = 20;
        const uint32_t dims = 1;
        const uint32_t label_dims = 1;
        
        const double LOW_LIMIT = -1;
        const double HI_LIMIT = 1;
	
        boost::shared_ptr<garf_types<feature_type>::matrix > x_ptr(new garf_types<feature_type>::matrix(num_training_samples,dims));
        garf_types<feature_type>::matrix& x = *x_ptr;
	    {
	        mt19937 gen;
	        uniform_real_distribution<> dimension_sampler(LOW_LIMIT, HI_LIMIT);
            
	        

            for (uint32_t i=0; i < num_training_samples; i++) {
                x(i, 0) = dimension_sampler(gen);
            }

        }
        //std::cout << "training on " << *m_p << std::endl;
        
        boost::shared_ptr<garf_types<label_type>::matrix > y_ptr(new garf_types<label_type>::matrix(num_training_samples, label_dims));
        y_ptr->clear();
        {
            matrix<label_type>& y = *y_ptr;
            
            mt19937 gen;
            normal_distribution<label_type> nd;
            
            for (uint32_t i=0; i < num_training_samples; i++) {
                for (uint32_t j=0; j < 1; j++) {
                    y(i,j) = x(i,j) * x(i,j);
                }
            }
        }
    
    
	    

    	boost::shared_ptr<regression_forest<feature_type, label_type, hyperplane_split_finder> > reg_f(new regression_forest<feature_type, label_type, hyperplane_split_finder>());
	    // the training set gets stored in a shared_ptr inside the forest as well, so keep like this.
        boost::shared_ptr<supervised_training_set<feature_type, label_type> > t_set(new multi_supervised_regression_training_set<feature_type, label_type>(x_ptr, y_ptr));
        
        std::cout << "about to do training, x = "  << *x_ptr << std::endl
            << "y = " << *y_ptr << std::endl << std::endl;
            
            
	    reg_f->train(t_set);

        t_set.reset(); // get rid of the reference to training set - it should *NOT* get deleted
        std::cout << "main shared pointer to training set deleted" << std::endl;
        
        
        // prediction - just put in training data for the moment
        garf_types<label_type>::matrix prediction_outputs(num_predict_samples, label_dims);
        std::cout << "predicting on " << *x_ptr << std::endl;
        reg_f->predict(*x_ptr, &prediction_outputs);
        
        std::cout << "prediction done, prediction_outputs = " << prediction_outputs << std::endl; 
        
    }
    catch(std::exception e) {
        std::cout << "Exception caught by user code: " << e.what() << std::endl;
    }


	
}