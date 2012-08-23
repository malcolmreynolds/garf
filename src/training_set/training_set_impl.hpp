#ifndef GARF_TRAINING_SET_IMPL_HPP
#define GARF_TRAINING_SET_IMPL_HPP

template<typename FeatureType>
training_set<FeatureType>::training_set(boost::shared_ptr<typename garf_types<FeatureType>::matrix> feature_matrix)
    : _feature_matrix(feature_matrix), _num_training_samples(feature_matrix->size1()), _feature_dimensionality(feature_matrix->size2()) {
}

#endif
