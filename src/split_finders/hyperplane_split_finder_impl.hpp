#ifndef GARF_HYPERPLANE_SPLIT_FINDER_IMPL_HPP
#define GARF_HYPERPLANE_SPLIT_FINDER_IMPL_HPP

#include <cmath>

namespace garf {
    
    template<typename FeatT, typename LabelT>
    hyperplane_split_finder<FeatT, LabelT>::hyperplane_split_finder(const training_set<FeatT> & training_set,
                                                                    const tset_indices_t & valid_indices,
                                                                    const multivariate_normal<LabelT> & parent_dist,
                                                                    training_set_idx_t num_in_parent,
                                                                    uint32_t num_splits_to_try,
                                                                    uint32_t num_threshes_per_split) 
        : split_finder<FeatT, LabelT>(training_set, valid_indices, parent_dist, num_in_parent, num_splits_to_try, num_threshes_per_split) {
        // The hyperplane dimensionality needs to be less than the full feature dimensionality
        // Let's say square root for the moment - round it up so that for 2 dim data we do 2d hyperplanes
       this->_hyperplane_dimensionality = ceil(sqrt(this->_feature_dimensionality));
#ifdef VERBOSE
        std::cout << "Hyperplane split finder for " << this->_feature_dimensionality
            << " dim data, going to gen " << this->_hyperplane_dimensionality
            << " dimensional hyperplanes" << std::endl;
#endif
    } 
        
    template<typename FeatT, typename LabelT>
    void hyperplane_split_finder<FeatT, LabelT>::calculate_all_features(const typename garf_types<feature_idx_t>::matrix feat_indices_mtx,
                                                                        const typename garf_types<FeatT>::matrix & plane_coeffs, 
                                                                        typename garf_types<FeatT>::matrix & feature_values) const {
        const typename garf_types<FeatT>::matrix& features = this->_training_set.features();
        split_idx_t num_planes_to_try = plane_coeffs.size1();
        training_set_idx_t num_data_points = this->_num_training_samples;
        
        // FIXME: this loop could surely be faster, maybe extract all the rows beforehand? Maybe dash the matrix_row thing
        // and just have explicit loop? If it wasn't for the 'valid_indices' stuff this would be a straight matrix product
        // of the feature matrix with the transpose of the plane matrix. As we get down the tree though we would only need 
        // soe of the output lines of that.
        for (split_idx_t p = 0; p < num_planes_to_try; p++) {
            // Make vector views of the feature indices and plane coefficients
            const matrix_row<const typename garf_types<feature_idx_t>::matrix> feat_indices(feat_indices_mtx, p);
            const matrix_row<const typename garf_types<FeatT>::matrix> plane(plane_coeffs, p);

            // Loop through training data
            for (training_set_idx_t i = 0; i < num_data_points; i++) {
                const matrix_row<const typename garf_types<FeatT>::matrix> datapoint(features, this->_valid_indices(i));
                FeatT sum = 0;
                // Now we need to do the inner product of only the selected features from our
                // datapoint, with each coefficient of the 
                for (feature_idx_t j=0; j < this->_hyperplane_dimensionality; j++) {
                    sum += (plane(j) * datapoint(feat_indices(j)));
                }
                feature_values(i, p) = sum;
            }
        }                                                                           
    }
        
    // Generate the hyperplane directions. Note that we don't use the feature_indices in this instance
    template<typename FeatT, typename LabelT>
    void hyperplane_split_finder<FeatT, LabelT>::generate_hyperplane_coefficients(typename garf_types<FeatT>::matrix & plane_coeffs,
                                                                                  const typename garf_types<feature_idx_t>::matrix & feature_indices,
                                                                                  boost::random::mt19937& gen) {
        boost::normal_distribution<> nd(0.0, 1.0);
        for (uint32_t i=0; i < this->num_splits_to_try(); i++) {
            for (uint32_t j=0; j < this->_hyperplane_dimensionality; j++) {
                plane_coeffs(i, j) = nd(gen);
            }
        }        
    }

    template<typename FeatT, typename LabelT>
    inf_gain_t hyperplane_split_finder<FeatT, LabelT>::find_optimal_split(tset_indices_t** samples_going_left,
                                                                    tset_indices_t** samples_going_right,
                                                                    multivariate_normal<LabelT>& left_dist, 
                                                                    multivariate_normal<LabelT>& right_dist,
                                                                    boost::random::mt19937& gen) {
        // Generate a matrix of indices for what features we care about -
        // might be better if some of these repeated.
        typename garf_types<feature_idx_t>::matrix feature_indices(this->num_splits_to_try(),
                                                                   this->_hyperplane_dimensionality);
        uniform_int_distribution<> idx_dist(0, (this->_feature_dimensionality - 1));
        for (uint32_t i=0; i < this->num_splits_to_try(); i++) {
            for (uint32_t j=0; j < this->_hyperplane_dimensionality; j++) {
                feature_indices(i, j) = idx_dist(gen);
            }
        }

        // Generate orientation of the hyperplanes
        typename garf_types<FeatT>::matrix plane_coeffs(this->num_splits_to_try(), 
                                                        this->_hyperplane_dimensionality);
#ifdef VERBOSE
        std::cout << "about to call generate_hyperplane_coefficients in " << this->name() << std::endl;
#endif 
        generate_hyperplane_coefficients(plane_coeffs, feature_indices, gen);

#ifdef VERBOSE
        std::cout << "feature dimensionality = " << this->_feature_dimensionality 
            << ", hyperplane dimensionality = " << this->_hyperplane_dimensionality
            << ", feature_indices = " << feature_indices
            << ", plane_coeffs = " << plane_coeffs << std::endl;
#endif

        //Evaluate each datapoint against each hyperplane
        typename garf_types<FeatT>::matrix feature_values(this->_num_training_samples, this->num_splits_to_try());
        calculate_all_features(feature_indices, plane_coeffs, feature_values);
#ifdef VERBOSE
        std::cout << "Evaluated all features:" << std::endl;
        print_matrix<FeatT>(std::cout, feature_values);
#endif

        inf_gain_t best_information_gain = this->pick_best_feature(feature_values,
                                                                   samples_going_left, samples_going_right,
                                                                   left_dist, right_dist, gen);
        matrix_row<typename garf_types<feature_idx_t>::matrix> best_feat_indices(feature_indices, this->best_split_idx());
        matrix_row<typename garf_types<FeatT>::matrix> best_plane(plane_coeffs, this->best_split_idx());
        _feature_subset.reset(new typename garf_types<feature_idx_t>::vector(this->_hyperplane_dimensionality));
        _feature_subset->assign(best_feat_indices);
        _plane_coeffs.reset(new typename garf_types<FeatT>::vector(this->_hyperplane_dimensionality));
        _plane_coeffs->assign(best_plane);

#ifdef VERBOSE                    
        std::cout << "Found best plane: features =" << *_feature_subset << std::endl;
        std::cout << "coeffs = " << *_plane_coeffs << std::endl;
        std::cout << "calculated information gains, result is:" << std::endl
            << (*samples_going_left)->size() << " go left: " << **samples_going_left  << std::endl
            << (*samples_going_right)->size() << " go right: " << **samples_going_right << std::endl;
#endif    
        return best_information_gain;
    }
    
    // Perform the inner product to see which side of the hyperplane we are on
    template<typename FeatT, typename LabelT>
    inline split_direction_t hyperplane_split_finder<FeatT, LabelT>::evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & row) const {
        double sum = 0.0;

        for (uint32_t i=0; i < this->_hyperplane_dimensionality; i++){
            sum += ((*_plane_coeffs)(i) * row((*_feature_subset)(i)));
        }

        if (sum <= this->best_split_thresh()) {
            return LEFT;
        }
        return RIGHT;
    }
}

#endif