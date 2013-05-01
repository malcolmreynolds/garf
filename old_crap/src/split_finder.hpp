#ifndef GARF_SPLIT_FINDER_HPP
#define GARF_SPLIT_FINDER_HPP

#include <boost/shared_ptr.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/nondet_random.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "forest_params.hpp"
#include "multivariate_normal.hpp"
#include "forest_utils.hpp"
#include "determinant.hpp"

#include <iostream>
#include <algorithm>
#include <float.h>
#include <cmath>

namespace garf {
        
//    using namespace boost::numeric::ublas;
    using namespace boost::random;
    
        
    // Thrown when for whatever reason (constant data) we can't formulate
    // any kind of split.
    struct split_error : std::runtime_error { 
        split_error() : std::runtime_error("Unable to find any useful splits at this node.") {}
    };
    
    template<typename FeatT, typename LabelT = FeatT> class split_finder; 
    template<typename FeatT, typename LabelT>
    class split_finder {
        // protected so we can access this all in the subclass

        // This is the *index* of the best split. In the case of axis aligned splitter this literally points
        // to the feature, and so nothing else is needed. In the case of the hyperplane, this is an index into the hyperplane
        // matrix and so some further cleaning up is necessary.
        split_idx_t _best_split_idx;
        FeatT _best_split_thresh; //value to split at. Assume we are always doing <= goes left, > goes right

        split_idx_t _num_splits_to_try;
        uint32_t _num_threshes_per_split;
        
    public:
        const training_set<FeatT> & _training_set;
        const tset_indices_t & _valid_indices;  // data indices present in the parent node which we are splitting
        const multivariate_normal<LabelT> & _parent_dist;
        const training_set_idx_t _num_in_parent;

        // Dimensionality of the features we are splitting on
        const feature_idx_t _feature_dimensionality;
        
        // Dimensionality of the data which we use to calculate information gains
        const inf_gain_t _inf_gain_dimensionality;
        training_set_idx_t _num_training_samples;

        // extra data needed for Manik Varma's tree balancing trick
        double _balance_bias;
        double _depth;

        virtual const char * name() const = 0;
        
        split_finder(const training_set<FeatT> & training_set,
                     const tset_indices_t & valid_indices,
                     const multivariate_normal<LabelT> & parent_dist,
                     training_set_idx_t num_in_parent,
                     split_idx_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~split_finder() {};
                     
        // Given feature matrix and a vector of indices into the feature matrix, return the left / right
        // decisions in a given output vector. Also return the distributions for the left and right children.
        // Returns the information gain that was found.
        virtual inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                              tset_indices_t** samples_going_right,
                                              multivariate_normal<LabelT>& left_dist, 
                                              multivariate_normal<LabelT>& right_dist,
                                              boost::random::mt19937& gen) = 0;
                                        
        // virtual void evaluate_split(training_set<FeatT>& training_set, 
        //                             const tset_indices_t& valid_samples,
        //                             garf_types<split_direction_t>::vector& split_direction) = 0;
        inline void set_num_splits_to_try(split_idx_t num_splits_to_try) { _num_splits_to_try = num_splits_to_try; };
        inline split_idx_t num_splits_to_try() const { return _num_splits_to_try; };

        inline void set_num_threshes_per_split(uint32_t num_threshes_per_split) { _num_threshes_per_split = num_threshes_per_split; };
        inline uint32_t num_threshes_per_split() const { return _num_threshes_per_split; };
        
        void calculate_mins_and_maxs(const typename garf_types<FeatT>::matrix& feature_values, 
                                     typename garf_types<FeatT>::vector& feature_mins,
                                     typename garf_types<FeatT>::vector& feature_maxs) const;
        
        void generate_thresholds(const typename garf_types<FeatT>::vector & feature_mins,
                                 const typename garf_types<FeatT>::vector & feature_maxs,
                                 typename garf_types<FeatT>::matrix& thresholds,
                                 boost::random::mt19937& gen) const;  
                                 
        inf_gain_t calculate_information_gains(const typename garf_types<FeatT>::matrix& thresholds,
                                         const typename garf_types<FeatT>::matrix& feature_values,
                                         tset_indices_t** samples_going_left,
                                         tset_indices_t** samples_going_right,
                                         multivariate_normal<LabelT>& left_dist_out, 
                                         multivariate_normal<LabelT>& right_dist_out);
        inf_gain_t pick_best_feature(const typename garf_types<FeatT>::matrix feature_values,
                               tset_indices_t** samples_going_left,
                               tset_indices_t** samples_going_right,
                               multivariate_normal<LabelT>& left_dist, 
                               multivariate_normal<LabelT>& right_dist,
                               boost::random::mt19937& gen);

        inline void evaluate_single_split(const typename garf_types<FeatT>::matrix& feature_values, 
                                   garf_types<split_direction_t>::vector& split_directions,
                                   feature_idx_t split_feature,
                                   FeatT split_value);
        
        // Evaluate the (trained) split for some piece of data
        inline virtual split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const = 0;

        inline void set_balance_bias(double balance_bias) { _balance_bias = balance_bias; }
        inline void set_depth(depth_t depth) { _depth = depth; }
        
        inline split_idx_t best_split_idx() const { return _best_split_idx; }
        inline void set_best_split_idx(split_idx_t new_best_split_idx) { _best_split_idx = new_best_split_idx; }

        inline FeatT best_split_thresh() const { return _best_split_thresh; };        
        inline void set_best_split_thresh(FeatT new_best_split_thresh) { _best_split_thresh = new_best_split_thresh; }

    };

    /* Call this class with paramters of how many dimensions and how many splits to try, then do 
       find_optimal_split() to pick the best one. */
    template<typename FeatT, typename LabelT>
    class axis_aligned_split_finder : public split_finder<FeatT, LabelT> {
        feature_idx_t _split_feature; // index of the feature we are splitting on

        void calculate_all_features(const typename garf_types<feature_idx_t>::vector & dims_to_try, 
                                    typename garf_types<FeatT>::matrix & feature_values) const;
        
    public:

        virtual const char * name() const { return "axis_aligned"; };

        axis_aligned_split_finder(const training_set<FeatT> & training_set,
                                  const tset_indices_t& valid_indices,
                                  const multivariate_normal<LabelT> & parent_dist,
                                  training_set_idx_t num_in_parent,
                                  uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~axis_aligned_split_finder() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                tset_indices_t** samples_going_right,
                                multivariate_normal<LabelT>& left_dist, 
                                multivariate_normal<LabelT>& right_dist,
                                boost::random::mt19937& gen);
        // Evaluate the split for a single piece of data
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;
#ifdef BUILD_PYTHON_BINDINGS
        inline feature_idx_t get_split_feature() const { return _split_feature; };
#endif
    };

    // will throw an error
    enum single_double_split_t {
        SINGLE,
        DOUBLE
    };
    template<typename FeatT, typename LabelT>
    class single_double_spltfndr : public split_finder<FeatT, LabelT> {
        single_double_split_t _type;
        feature_idx_t _feat_i; // indices to the features we are going to do hyper
        feature_idx_t _feat_j;

        FeatT _coeff_i; // multiply the feature values by this to project into the hyperplane
        FeatT _coeff_j; 

        void calculate_singular_features(const typename garf_types<feature_idx_t>::vector & dims_to_try, 
                                         typename garf_types<FeatT>::matrix & feature_values) const;
        void calculate_pairwise_features(const typename garf_types<feature_idx_t>::matrix & pairwise_dims,
                                         const typename garf_types<FeatT>::matrix & pairwise_coeffs,
                                         typename garf_types<FeatT>::matrix & feature_values,
                                         split_idx_t offset) const;
    public:
        virtual const char * name() const { return "single_double"; };
        
        single_double_spltfndr(const training_set<FeatT> & training_set,
                               const tset_indices_t& valid_indices,
                               const multivariate_normal<LabelT> & parent_dist,
                               training_set_idx_t num_in_parent,
                               uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~single_double_spltfndr() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                           tset_indices_t** samples_going_right,
                           multivariate_normal<LabelT>& left_dist, 
                           multivariate_normal<LabelT>& right_dist,
                           boost::random::mt19937& gen);

        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;

#ifdef BUILD_PYTHON_BINDINGS
        inline feature_idx_t get_feat_i() const { return _feat_i; }
        inline feature_idx_t get_feat_j() const { return _feat_j; }
        inline FeatT get_coeff_i() const { return _coeff_i; }
        inline FeatT get_coeff_j() const { return _coeff_j; }
        inline single_double_split_t get_type() const { return _type; }
#endif
    };
   

    template<typename FeatT, typename LabelT>
    class hyperplane_split_finder : public split_finder<FeatT, LabelT> {
        feature_idx_t _hyperplane_dimensionality;

        // Store the indices of features which we compare to
        boost::shared_ptr<typename garf_types<feature_idx_t>::vector> _feature_subset;

        // Store the coefficients which define the hyperplane.
        boost::shared_ptr<typename garf_types<FeatT>::vector> _plane_coeffs;
       
        void calculate_all_features(const typename garf_types<feature_idx_t>::matrix feat_indices_mtx,
                                    const typename garf_types<FeatT>::matrix & plane_coeffs,
                                    typename garf_types<FeatT>::matrix & feature_values) const;
                                   
    public:
        virtual const char * name() const { return "hyperplane"; };
        virtual void generate_hyperplane_coefficients(typename garf_types<FeatT>::matrix & plane_coeffs,
                                              const typename garf_types<feature_idx_t>::matrix & feature_indices,
                                              boost::random::mt19937& gen);
        hyperplane_split_finder(const training_set<FeatT> & training_set,
                                const tset_indices_t& valid_indices,
                                const multivariate_normal<LabelT> & parent_dist,
                                training_set_idx_t num_in_parent,
                                uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~hyperplane_split_finder() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                tset_indices_t** samples_going_right,
                                multivariate_normal<LabelT>& left_dist,
                                multivariate_normal<LabelT>& right_dist,
                                boost::random::mt19937& gen);
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;

        inline feature_idx_t get_hyperplane_dimensionality() const { return _hyperplane_dimensionality; }

#ifdef BUILD_PYTHON_BINDINGS
        inline typename garf_types<feature_idx_t>::vector get_feature_subset() const { return *_feature_subset; }
        inline typename garf_types<FeatT>::vector get_plane_coeffs() const { return *_plane_coeffs; }
#endif
    };

    // As above, except we sample the locations for the hyperplanes by taking two pieces of data building the normal
    // between them
    template<typename FeatT, typename LabelT>
    class smart_hyperplane_split_finder : public hyperplane_split_finder<FeatT, LabelT> {
    public:
        smart_hyperplane_split_finder(const training_set<FeatT> & training_set,
                                      const tset_indices_t& valid_indices,
                                      const multivariate_normal<LabelT> & parent_dist,
                                      training_set_idx_t num_in_parent,
                                      uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual const char * name() const { return "smart_hyperplane"; };

        virtual void generate_hyperplane_coefficients(typename garf_types<FeatT>::matrix & plane_coeffs,
                                              const typename garf_types<feature_idx_t>::matrix & feature_indices,
                                              boost::random::mt19937& gen);
    };

    // FIXME: could combine some common code between the two pairwise split classes
    template<typename FeatT, typename LabelT>
    class pairwise_diff_split_finder : public split_finder<FeatT, LabelT> {
        // Our feature is made by thresholding the difference between raw features i and j
        feature_idx_t _feat_i;
        feature_idx_t _feat_j;
        
        void calculate_all_features(const typename garf_types<feature_idx_t>::vector & feat_i_candidates,
                                    const typename garf_types<feature_idx_t>::vector & feat_j_candidates,
                                    typename garf_types<FeatT>::matrix & feature_values) const;
    public:
        virtual const char * name() const { return "pairwise_diff"; };
        
        pairwise_diff_split_finder(const training_set<FeatT> & training_set,
                                   const tset_indices_t& valid_indices,
                                   const multivariate_normal<LabelT> & parent_dist,
                                   training_set_idx_t num_in_parent,
                                   uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~pairwise_diff_split_finder() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                tset_indices_t** samples_going_right,
                                multivariate_normal<LabelT> & left_dist,
                                multivariate_normal<LabelT> & right_dist,
                                boost::random::mt19937 & gen);
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;
#ifdef BUILD_PYTHON_BINDINGS
        inline feature_idx_t get_feat_i() const { return _feat_i; };
        inline feature_idx_t get_feat_j() const { return _feat_j; };
#endif
    };

    // This is for the resolution split of 2x2, 4x4, 8x8, 16x16 - anything else
    // will throw an error
    template<typename FeatT, typename LabelT>
    class ctf_pairwise_only_split_finder : public split_finder<FeatT, LabelT> {
        feature_idx_t _feat_i;
        feature_idx_t _feat_j;

 
        
        void calculate_all_features(const typename garf_types<feature_idx_t>::vector & feat_i_candidates,
                                    const typename garf_types<feature_idx_t>::vector & feat_j_candidates,
                                    typename garf_types<FeatT>::matrix & feature_values) const;
                                    
    public:
        virtual const char * name() const { return "ctf_pairwise_only"; };

        ctf_pairwise_only_split_finder(const training_set<FeatT> & training_set,
                                             const tset_indices_t& valid_indices,
                                             const multivariate_normal<LabelT> & parent_dist,
                                             training_set_idx_t num_in_parent,
                                             uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~ctf_pairwise_only_split_finder() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                tset_indices_t** samples_going_right,
                                multivariate_normal<LabelT> & left_dist,
                                multivariate_normal<LabelT> & right_dist,
                                boost::random::mt19937 & gen);                                       
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;
#ifdef BUILD_PYTHON_BINDINGS
        inline feature_idx_t get_feat_i() const { return _feat_i; };
        inline feature_idx_t get_feat_j() const { return _feat_j; };
#endif
    };

    // This is for the resolution split of 2x2, 4x4, 8x8, 16x16 - anything else
    // will throw an error
    enum split_t {
        PAIR_ABS,
        PAIR_DIFF,
        PAIR_PLUS,
        SINGULAR
    };
    template<typename FeatT, typename LabelT>
    class ctf_pairwise_singular_split_finder : public split_finder<FeatT, LabelT> {
        feature_idx_t _feat_i;
        feature_idx_t _feat_j;
        split_t _type;

        void calculate_pairwise_diff_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
                                          const garf_types<feature_idx_t>::vector & feat_j_candidates,
                                          typename garf_types<FeatT>::matrix & feature_values,
                                          uint32_t feat_val_offset) const;

        void calculate_pairwise_plus_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
                                          const garf_types<feature_idx_t>::vector & feat_j_candidates,
                                          typename garf_types<FeatT>::matrix & feature_values,
                                          uint32_t feat_val_offset) const;

        void calculate_pairwise_abs_features(const garf_types<feature_idx_t>::vector & feat_i_candidates,
                                         const garf_types<feature_idx_t>::vector & feat_j_candidates,
                                         typename garf_types<FeatT>::matrix & feature_values,
                                         uint32_t feat_val_offset) const;                

        void calculate_singular_features(const typename garf_types<feature_idx_t>::vector & singular_feat_candidates,
                                         typename garf_types<FeatT>::matrix & feature_values,
                                         uint32_t feat_val_offset) const;


        
                                    
    public:
        virtual const char * name() const { return "ctf_pairwise_singular"; };
        ctf_pairwise_singular_split_finder(const training_set<FeatT> & training_set,
                                             const tset_indices_t& valid_indices,
                                             const multivariate_normal<LabelT> & parent_dist,
                                             training_set_idx_t num_in_parent,
                                             uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~ctf_pairwise_singular_split_finder() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                tset_indices_t** samples_going_right,
                                multivariate_normal<LabelT> & left_dist,
                                multivariate_normal<LabelT> & right_dist,
                                boost::random::mt19937 & gen);                                       
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;
#ifdef BUILD_PYTHON_BINDINGS
        inline bool get_type() const { return _type; };
        inline feature_idx_t get_feat_i() const { return _feat_i; };
        inline feature_idx_t get_feat_j() const { return _feat_j; };
#endif
    };

    /* 
        Pick uniformly between levels, then select two ncc vectors and split on dot product
    */
    template<typename FeatT, typename LabelT>
    class ncc_levels_2468_pairwise_spltfndr : public split_finder<FeatT, LabelT> {
        // Indices to the x components of the two vectors. y components are simply
        // the one after in memory
        feature_idx_t _feat_i_x;
        // feature_idx_t _feat_i_y;
        feature_idx_t _feat_j_x; 
        // feature_idx_t _feat_j_y;

        void calculate_vec_dot_prod_features(const garf_types<feature_idx_t>::vector & feat_1_candidates,
                                             const garf_types<feature_idx_t>::vector & feat_2_candidates,
                                             typename garf_types<FeatT>::matrix & feature_values) const;


    public:
        virtual const char * name() const { return "ncc_levels_2468_pairwise"; };
        
        ncc_levels_2468_pairwise_spltfndr(const training_set<FeatT> & training_set,
                                           const tset_indices_t & valid_indices,
                                           const multivariate_normal<LabelT> & parent_dist,
                                           training_set_idx_t num_in_parent,
                                           uint32_t num_splits_to_try, uint32_t num_threshes_per_split);
        virtual ~ncc_levels_2468_pairwise_spltfndr() {};
        inf_gain_t find_optimal_split(tset_indices_t** samples_going_left,
                                      tset_indices_t** samples_going_right,
                                      multivariate_normal<LabelT> & left_dist,
                                      multivariate_normal<LabelT> & right_dist,
                                      boost::random::mt19937 & gen);
        inline split_direction_t evaluate(const matrix_row<const typename garf_types<FeatT>::matrix> & sample) const;
#ifdef BUILD_PYTHON_BINDINGS
        inline feature_idx_t get_feat_i_x() const { return _feat_i_x; }
        inline feature_idx_t get_feat_j_x() const { return _feat_j_x; }
#endif

    };

}

#include "split_finders/split_finder_impl.hpp"
#include "split_finders/axis_aligned_split_finder_impl.hpp"
#include "split_finders/hyperplane_split_finder_impl.hpp"
#include "split_finders/smart_hyperplane_split_finder_impl.hpp"
#include "split_finders/pairwise_diff_split_finder_impl.hpp"
#include "split_finders/ctf_pairwise_only_split_finder_impl.hpp"
#include "split_finders/ctf_pairwise_singular_split_finder_impl.hpp"
#include "split_finders/ncc_levels_2468_pairwise_spltfndr_impl.hpp"
#include "split_finders/single_double_spltfndr_impl.hpp"

#endif