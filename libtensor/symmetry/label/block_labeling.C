#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "block_labeling.h"
#include "block_labeling_impl.h"


namespace libtensor {

template class block_labeling<1>;
template class block_labeling<2>;
template class block_labeling<3>;
template class block_labeling<4>;
template class block_labeling<5>;
template class block_labeling<6>;

template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<1> &,
        const sequence<1, size_t> &, block_labeling<6> &);

template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<2> &,
        const sequence<2, size_t> &, block_labeling<6> &);

template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<3> &,
        const sequence<3, size_t> &, block_labeling<6> &);

template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<4> &,
        const sequence<4, size_t> &, block_labeling<6> &);

template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<5> &,
        const sequence<5, size_t> &, block_labeling<6> &);

template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<1> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<2> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<3> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<5> &);
template void transfer_labeling(const block_labeling<6> &,
        const sequence<6, size_t> &, block_labeling<6> &);

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

