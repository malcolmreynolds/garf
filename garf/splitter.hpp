#ifndef GARF_SPLITTER_HPP
#define GARF_SPLITTER_HPP

#include "types.hpp"

#ifdef GARF_SERIALIZE_ENABLE
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#endif


namespace garf {


    const double NaN = std::numeric_limits<double>::quiet_NaN();

    // Simply the parent class for all other splitters
    class Splitter {
        // Don't really need anything in here. We used to have a virtual function
        // for evaluate() but I actually want it inlined everywhere, and also we are
        // always going to be working with specific instances of subclasses (thanks
        // templates)
    };

    // Splitter which does looks at a single feature dimension, tests against a single threshold
    template<typename FeatT>
    class AxisAlignedSplt : public Splitter {
    public:
        feat_idx_t feat_idx;
        FeatT thresh;
        inline split_dir_t evaluate(const feature_vec<FeatT> & fvec) const {
            if (fvec(feat_idx) <= thresh) {
                return LEFT;
            }
            return RIGHT;
        }
        // Initialise to invalid values (-1, NaN) so we know if we are using uninitialised data
        AxisAlignedSplt() : feat_idx(-1), thresh(NaN) {} 
        inline char const * name() const { return "axis_aligned"; }

        template<typename T>
        inline friend std::ostream& operator<< (std::ostream& stream, const AxisAlignedSplt<T>& aas);
#ifdef GARF_SERIALIZE_ENABLE
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & feat_idx;
            ar & thresh;
        }
#endif
    };

    // splitter which looks at two dimensions scaled by two weights, compares to a single threshold
    template<typename FeatT>
    class TwoDimSplt : public Splitter {
    public:
        feat_idx_t feat_1;
        feat_idx_t feat_2;
        double weight_feat_1;
        double weight_feat_2;
        FeatT thresh;
        inline split_dir_t evaluate(const feature_vec<FeatT> & fvec) const {
            double test_val = (fvec(feat_1) * weight_feat_1) + 
                              (fvec(feat_2) * weight_feat_2);
            if (test_val <= thresh) {
                return LEFT;
            }
            return RIGHT;
        }
        TwoDimSplt(): feat_1(-1), feat_2(-1), weight_feat_1(NaN), weight_feat_2(NaN), thresh(NaN) {}
        inline char const * name() const { return "2_dim_hyp"; }

        template<typename T>
        inline friend std::ostream& operator<< (std::ostream& stream, const TwoDimSplt<T>& two_ds);
#ifdef GARF_SERIALIZE_ENABLE
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & feat_1;
            ar & feat_2;
            ar & weight_feat_1;
            ar & weight_feat_2;
            ar & thresh;
        }
#endif
    };
}

#include "splitter_print.cpp"

#endif