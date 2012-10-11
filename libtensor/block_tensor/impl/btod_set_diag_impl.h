#ifndef LIBTENSOR_BTOD_SET_DIAG_IMPL_H
#define LIBTENSOR_BTOD_SET_DIAG_IMPL_H

#include "../btod_set_diag.h"

namespace libtensor {


template<size_t N>
const char *btod_set_diag<N>::k_clazz = "btod_set_diag<N>";


template<size_t N>
btod_set_diag<N>::btod_set_diag(double v) : m_gbto(v) {

}


template<size_t N>
void btod_set_diag<N>::perform(block_tensor_i<N, double> &bt) {

    m_gbto.perform(bt);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_IMPL_H
