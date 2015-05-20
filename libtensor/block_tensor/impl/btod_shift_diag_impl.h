#ifndef LIBTENSOR_BTOD_SHIFT_DIAG_IMPL_H
#define LIBTENSOR_BTOD_SHIFT_DIAG_IMPL_H

#include "../btod_shift_diag.h"

namespace libtensor {


template<size_t N>
const char *btod_shift_diag<N>::k_clazz = "btod_shift_diag<N>";


template<size_t N>
btod_shift_diag<N>::btod_shift_diag(const sequence<N, size_t> &msk, double v) :
    m_gbto(msk, v) {

}


template<size_t N>
void btod_shift_diag<N>::perform(block_tensor_i<N, double> &bt) {

    m_gbto.perform(bt);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SHIFT_DIAG_IMPL_H
