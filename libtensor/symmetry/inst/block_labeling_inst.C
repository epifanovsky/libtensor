#include <libtensor/btod/scalar_transf_double.h>
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
template class block_labeling<13>;
template class block_labeling<14>;
template class block_labeling<15>;
template class block_labeling<16>;

template
bool operator==(const block_labeling<1>&, const block_labeling<1>&);
template
bool operator==(const block_labeling<2>&, const block_labeling<2>&);
template
bool operator==(const block_labeling<3>&, const block_labeling<3>&);
template
bool operator==(const block_labeling<4>&, const block_labeling<4>&);
template
bool operator==(const block_labeling<5>&, const block_labeling<5>&);
template
bool operator==(const block_labeling<6>&, const block_labeling<6>&);
template
bool operator==(const block_labeling<7>&, const block_labeling<7>&);
template
bool operator==(const block_labeling<8>&, const block_labeling<8>&);
template
bool operator==(const block_labeling<9>&, const block_labeling<9>&);
template
bool operator==(const block_labeling<10>&, const block_labeling<10>&);
template
bool operator==(const block_labeling<11>&, const block_labeling<11>&);
template
bool operator==(const block_labeling<12>&, const block_labeling<12>&);
template
bool operator==(const block_labeling<13>&, const block_labeling<13>&);
template
bool operator==(const block_labeling<14>&, const block_labeling<14>&);
template
bool operator==(const block_labeling<15>&, const block_labeling<15>&);
template
bool operator==(const block_labeling<16>&, const block_labeling<16>&);

template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<1>&,
    const sequence<1, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<2>&,
    const sequence<2, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<3>&,
    const sequence<3, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<4>&,
    const sequence<4, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<5>&,
    const sequence<5, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<6>&,
    const sequence<6, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<7>&,
    const sequence<7, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<8>&,
    const sequence<8, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<9>&,
    const sequence<9, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<10>&,
    const sequence<10, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<11>&,
    const sequence<11, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<12>&,
    const sequence<12, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<13>&,
    const sequence<13, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<14>&,
    const sequence<14, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<15>&,
    const sequence<15, size_t>&, block_labeling<16>&);

template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<1>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<2>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<3>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<4>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<5>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<6>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<7>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<8>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<9>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<10>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<11>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<12>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<13>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<14>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<15>&);
template void transfer_labeling(const block_labeling<16>&,
    const sequence<16, size_t>&, block_labeling<16>&);


} // namespace libtensor
