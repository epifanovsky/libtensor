#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../block_labeling.h"
#include "block_labeling_impl.h"


namespace libtensor {

template class block_labeling<1>;
template class block_labeling<2>;
template class block_labeling<3>;
template class block_labeling<4>;
template class block_labeling<5>;
template class block_labeling<6>;
template class block_labeling<7>;
template class block_labeling<8>;
template class block_labeling<9>;
template class block_labeling<10>;
template class block_labeling<11>;
template class block_labeling<12>;

extern template
bool operator==(const block_labeling<1> &, const block_labeling<1> &);
extern template
bool operator==(const block_labeling<2> &, const block_labeling<2> &);
extern template
bool operator==(const block_labeling<3> &, const block_labeling<3> &);
extern template
bool operator==(const block_labeling<4> &, const block_labeling<4> &);
extern template
bool operator==(const block_labeling<5> &, const block_labeling<5> &);
extern template
bool operator==(const block_labeling<6> &, const block_labeling<6> &);
extern template
bool operator==(const block_labeling<7> &, const block_labeling<7> &);
extern template
bool operator==(const block_labeling<8> &, const block_labeling<8> &);
extern template
bool operator==(const block_labeling<9> &, const block_labeling<9> &);
extern template
bool operator==(const block_labeling<10> &, const block_labeling<10> &);
extern template
bool operator==(const block_labeling<11> &, const block_labeling<11> &);
extern template
bool operator==(const block_labeling<12> &, const block_labeling<12> &);

template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<7> &,
        const sequence<7, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<8> &,
        const sequence<8, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<9> &,
        const sequence<9, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<10> &,
        const sequence<10, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<11> &,
        const sequence<11, size_t> &, block_labeling<12> &);

template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<4> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<6> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<7> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<8> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<9> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<10> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<11> &);
template void transfer_labeling(const block_labeling<12> &,
        const sequence<12, size_t> &, block_labeling<12> &);

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

