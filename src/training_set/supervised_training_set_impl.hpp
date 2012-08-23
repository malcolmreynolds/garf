#ifndef GARF_SUPERVISED_TRAINING_SET_IMPL_HPP
#define GARF_SUPERVISED_TRAINING_SET_IMPL_HPP

template<typename FeatT, typename LabelT>
supervised_training_set<FeatT, LabelT>::supervised_training_set(boost::shared_ptr<typename garf_types<FeatT>::matrix> feature_matrix) 
    : training_set<FeatT>(feature_matrix) { 
}

#endif