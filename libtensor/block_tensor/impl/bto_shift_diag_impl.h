#ifndef LIBTENSOR_BTO_SHIFT_DIAG_IMPL_H
#define LIBTENSOR_BTO_SHIFT_DIAG_IMPL_H

#include "../bto_shift_diag.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_shift_diag<N, T>::k_clazz = "bto_shift_diag<N, T>";


template<size_t N, typename T>
bto_shift_diag<N, T>::bto_shift_diag(const sequence<N, size_t> &msk, T v) :
    m_gbto(msk, v) {

}


template<size_t N, typename T>
void bto_shift_diag<N, T>::perform(block_tensor_i<N, T> &bt) {

    m_gbto.perform(bt);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SHIFT_DIAG_IMPL_H
