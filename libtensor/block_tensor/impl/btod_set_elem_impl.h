#ifndef LIBTENSOR_BTOD_SET_ELEM_IMPL_H
#define LIBTENSOR_BTOD_SET_ELEM_IMPL_H

#include "../btod_set_elem.h"

namespace libtensor {


template<size_t N>
const char *btod_set_elem<N>::k_clazz = "btod_set_elem<N>";


template<size_t N>
void btod_set_elem<N>::perform(block_tensor_i<N, double> &bt,
    const index<N> &bidx, const index<N> &idx, double d) {

    m_gbto.perform(bt, bidx, idx, d);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_ELEM_IMPL_H
