#include <vector>

#include "forest_utils.hpp"

namespace garf {
    // Counts the number of elements in split_directions which match the direction dir
    training_set_idx_t count_elements_matching(const garf_types<split_direction_t>::vector& split_directions, split_direction_t dir) {
        
        training_set_idx_t num_matching = 0;
        // go through the vector with iterators, this appears to be best practice...
        garf_types<split_direction_t>::vector::const_iterator beginning = split_directions.begin();
        garf_types<split_direction_t>::vector::const_iterator end = split_directions.end();
        
        for (garf_types<split_direction_t>::vector::const_iterator it = beginning; it != end; it++) {
            if (*it == dir) {
                num_matching++;
            }
        }
        return num_matching;
    }
    
    // fills up sample_vec_out with all the indices from all_sample_vec_in where the corresponding direction in
    // split_directison matches dir. Example: if split_directions is [LEFT, LEFT, RIGHT, LEFT, RIGHT] and all_sample_vec_in
    // is [0, 1, 2, 3, 4] then sample_vec_out_left should be [0, 1, 3] and sample_vec_out_right should be [2,4]
    void fill_sample_vectors(garf_types<training_set_idx_t>::vector& sample_vec_out_left,
                             garf_types<training_set_idx_t>::vector& sample_vec_out_right, 
                             const garf_types<split_direction_t>::vector& split_directions,
                             const garf_types<training_set_idx_t>::vector& all_sample_vec_in) {
        
        garf_types<split_direction_t>::vector::size_type total_num = split_directions.size();
        if (total_num != all_sample_vec_in.size()) {
            throw new std::invalid_argument("all_sample_vec_in->size() and split_directions_in->size() must be equal");
        }

        garf_types<training_set_idx_t>::vector::size_type l_idx = 0;
        garf_types<training_set_idx_t>::vector::size_type r_idx = 0;
        split_direction_t dir;
        for (garf_types<split_direction_t>::vector::size_type i=0; i < total_num; i++) {
            dir = split_directions(i);
            switch(dir) {
            case LEFT:
                sample_vec_out_left(l_idx) = all_sample_vec_in(i);
                l_idx++;
                break;
            case RIGHT:
                sample_vec_out_right(r_idx) = all_sample_vec_in(i);
                r_idx++;
                break;
            default:
                throw new std::logic_error("Should never reach default in this switch statement.");
            }
        }   
    }
    


}