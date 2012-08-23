#ifndef GARF_TRAINING_SET_HPP
#define GARF_TRAINING_SET_HPP

#include <boost/shared_ptr.hpp>

#include "types.hpp"
#include "multivariate_normal.hpp"

#include <iostream>

namespace garf {
    
    // This is the parent class of the training_set hierarchy. It both stores the data for a forest, and allows
    // the forest to calculate information gains used for node splitting. This is in one class because the type
    // of data and the method of calculating information gain is inextricably linked.
    template<typename FeatT>
    class training_set {
        
        // the features x - common to all types of training sets
        boost::shared_ptr<typename garf_types<FeatT>::matrix> _feature_matrix;
        
        // The two dimensions of the feature matrix
        const training_set_idx_t _num_training_samples;
        const feature_idx_t _feature_dimensionality;
                
    public:
        training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix);
#ifdef VERBOSE        
        inline ~training_set() { std::cout << "~training_set()"<< std::endl; };
#endif 
        inline feature_idx_t feature_dimensionality () const { return _feature_dimensionality; };
        inline training_set_idx_t num_training_samples () const { return _num_training_samples; };
        inline const typename garf_types<FeatT>::matrix & features () const { return *_feature_matrix; };
        virtual const typename garf_types<FeatT>::matrix& dist_features() const { return *_feature_matrix; };

        virtual inf_gain_t information_gain(const multivariate_normal<FeatT> & parent_dist,
                                            const tset_indices_t& left_indices,
                                            const tset_indices_t& right_indices,
                                            const training_set_idx_t num_in_parent,
                                            const training_set_idx_t num_going_left,
                                            const training_set_idx_t num_going_right,
                                            multivariate_normal<FeatT> * temp_dist_l,
                                            multivariate_normal<FeatT> * temp_dist_r) const = 0;

        // Dimensionality of the data we do information gain on (ie the data that we need to fit gaussians to)
        virtual feature_idx_t inf_gain_dimensionality() const = 0;
    };
    
    // Training set with no labels
    template<typename FeatT>
    class unsupervised_training_set : public training_set<FeatT> {
    public:
        unsupervised_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix);
        virtual ~unsupervised_training_set() { std::cout << "~unsupervised_training_set()" << std::endl; };
        virtual inf_gain_t information_gain(const multivariate_normal<FeatT> & parent_dist,
                                            const tset_indices_t & left_indices,
                                            const tset_indices_t & right_indices,
                                            const training_set_idx_t num_in_parent,
                                            const training_set_idx_t num_going_left,
                                            const training_set_idx_t num_going_right,
                                            multivariate_normal<FeatT> * temp_dist_l,
                                            multivariate_normal<FeatT> * temp_dist_r) const;
        // virtual void set_parent_indices(const tset_indices_t& parent_indices);
        // virtual void clear_parent_indices();
        virtual feature_idx_t inf_gain_dimensionality() const { return this->feature_dimensionality(); };
    };
    
    // This is just an 'intermediate class' so the regression forest can take in one of these
    // and it doesn't matter whether we are dealing with 1d or multi-d regression
    template<typename FeatT, typename LabelT>
    class supervised_training_set : public training_set<FeatT> {
    public:
        supervised_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix);
        virtual label_idx_t label_dimensionality() const = 0;
    };
    
    // Training set with multi-dimensional labels
    template<typename FeatT, typename LabelT>
    class multi_supervised_regression_training_set : public supervised_training_set<FeatT, LabelT> {
        // store the labels
        boost::shared_ptr<typename garf_types<LabelT>::matrix> _labels;
        label_idx_t _label_dimensionality;
        
    public:
        multi_supervised_regression_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix,
                                                 boost::shared_ptr<typename garf_types<LabelT>::matrix> labels);

        virtual ~multi_supervised_regression_training_set() { std::cout << "~multi_supervised_regression_training_set()" << std::endl; };

        virtual inline label_idx_t label_dimensionality() const { return _label_dimensionality; };
        inline void hello() { std::cout << "hello from multi_supervised_regression_training_set" << std::endl; };
        virtual inf_gain_t information_gain(const multivariate_normal<LabelT> & parent_dist,
                                            const tset_indices_t & left_indices,
                                            const tset_indices_t & right_indices,
                                            const training_set_idx_t num_in_parent,
                                            const training_set_idx_t num_going_left,
                                            const training_set_idx_t num_going_right,
                                            multivariate_normal<FeatT> * temp_dist_l,
                                            multivariate_normal<FeatT> * temp_dist_r) const;
        // virtual void set_parent_indices(const tset_indices_t& parent_indices);
        // virtual void clear_parent_indices();
        virtual feature_idx_t inf_gain_dimensionality() const { return (feature_idx_t)_label_dimensionality; };
        virtual const typename garf_types<FeatT>::matrix& dist_features() const { return *_labels; };
        
    };
    
    
    //implementations in separate files
    #include "training_set/training_set_impl.hpp"
    #include "training_set/unsupervised_training_set_impl.hpp"
    #include "training_set/supervised_training_set_impl.hpp"
    #include "training_set/multi_supervised_regression_training_set_impl.hpp"
    
}


#endif