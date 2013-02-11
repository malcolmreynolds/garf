namespace garf {

	template<typename FeatT>
    inline std::ostream& operator<< (std::ostream& stream, const AxisAlignedSplt<FeatT>& aas) {
        stream << "[AxAlign[" << aas.feat_idx << ":" << aas.thresh << "]]";
        return stream;
    }

    template<typename FeatT>
    inline std::ostream& operator<< (std::ostream& stream, const TwoDimSplt<FeatT>& two_ds) {
    	stream << "[2Dim[" << two_ds.feat_1 << "*" << two_ds.weight_feat_1 << ","
    		               << two_ds.feat_2 << "*" << two_ds.weight_feat_2 << "]]";
		return stream;
    }

}