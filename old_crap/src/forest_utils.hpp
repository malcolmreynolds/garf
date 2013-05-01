#ifndef GARF_FOREST_UTILS_HPP
#define GARF_FOREST_UTILS_HPP

#include <boost/numeric/ublas/matrix.hpp>

#include "types.hpp"
//#include "manifold_forest_params.hpp"

#include <iostream>

namespace garf {

    // Counts the number of elements in split_directions which match the direction dir
    training_set_idx_t count_elements_matching(const garf_types<split_direction_t>::vector& split_directions, split_direction_t dir);

    // fills up sample_vec_out with all the indices from all_sample_vec_in where the corresponding direction in
    // split_directison matches dir
    void fill_sample_vectors(garf_types<training_set_idx_t>::vector& sample_vec_out_left,
                             garf_types<training_set_idx_t>::vector& sample_vec_out_right, 
                             const garf_types<split_direction_t>::vector& split_directions,
                             const garf_types<training_set_idx_t>::vector& all_sample_vec_in);

                     
    template<typename FeatureType>
    void print_matrix_selected_indices(std::ostream& out, const typename garf_types<FeatureType>::matrix& m,
                                       const garf_types<training_set_idx_t>::vector& indices) {
        
        typename garf_types<FeatureType>::matrix::size_type num_data_points = indices.size();           
        typename garf_types<FeatureType>::matrix::size_type dimensions = m.size2();                 
        for (typename garf_types<FeatureType>::matrix::size_type i=0; i < num_data_points; i++) {
            training_set_idx_t idx = indices(i);
            for (typename garf_types<FeatureType>::matrix::size_type dim_idx=0; dim_idx < dimensions; dim_idx++) {
                out << m(idx, dim_idx) << " ";
            }
            out << std::endl;
        }
    }
    
    // As above, but used when only part of the valid_indices vector is valid.
    template<typename FeatureType>
    void print_matrix_selected_indices_limit(std::ostream& out, const typename garf_types<FeatureType>::matrix& m,
                                       const garf_types<training_set_idx_t>::vector& indices, 
                                       uint32_t num_data_points) {
        
        typename garf_types<FeatureType>::matrix::size_type dimensions = m.size2();                 
        for (typename garf_types<FeatureType>::matrix::size_type i=0; i < num_data_points; i++) {
            training_set_idx_t idx = indices(i);
            for (typename garf_types<FeatureType>::matrix::size_type dim_idx=0; dim_idx < dimensions; dim_idx++) {
                out << m(idx, dim_idx) << " ";
            }
            out << std::endl;
        }
    }
    
    template<typename FeatureType>
    void print_matrix(std::ostream& out, const typename garf_types<FeatureType>::matrix& m) {
        typename garf_types<FeatureType>::matrix::size_type num_data_points = m.size1();
        typename garf_types<FeatureType>::matrix::size_type num_dimensions = m.size2();
        for (typename garf_types<FeatureType>::matrix::size_type i=0; i < num_data_points; i++) {
            for (typename garf_types<FeatureType>::matrix::size_type dim_idx = 0; dim_idx < num_dimensions; dim_idx++) {
                out << m(i, dim_idx) << " ";
            }
            out << std::endl;
        }
    }
    
    template<typename T>
    void print_vector_up_to(std::ostream& out, const typename garf_types<T>::vector& v, typename garf_types<T>::vector::size_type n) {
        for (typename garf_types<T>::vector::size_type i = 0; i < n; i++) {
            out << v(i) << " ";
        }
        out << std::endl;
    }
    
    // template<typename T>
    // void print_shared_ptr_refcount(boost::shared_ptr<T> p, std::ostream& out = std::cout) {
    //     out << "p=" << p << " px=" << p.px << " use count=" << p.pn.pi_->use_count << " weak count=" << p.pn.pi_->weak_count << std::endl;
    // }
}

#endif