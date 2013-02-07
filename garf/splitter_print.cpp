namespace garf {

    inline std::ostream& operator<< (std::ostream& stream, const AxisAlignedSplt& aas) {
        stream << "[AxAlign[" << aas.feat_idx << ":" << aas.thresh << "]]";
        return stream;
    }

    inline std::ostream& operator<< (std::ostream& stream, const TwoDimSplt& two_ds) {
    	stream << "[2Dim[" << two_ds.feat_1 << "*" << two_ds.weight_feat_1 << ","
    		               << two_ds.feat_2 << "*" << two_ds.weight_feat_2 << "]]";
		return stream;
    }

}