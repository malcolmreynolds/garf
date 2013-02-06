namespace garf {

    inline std::ostream& operator<< (std::ostream& stream, const AxisAlignedSplt& aas) {
        stream << "[AxAligned[" << aas.feat_idx << ":" << aas.thresh << "]]";
        return stream;
    }

}