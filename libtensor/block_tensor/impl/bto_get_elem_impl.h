#ifndef LIBTENSOR_BTO_GET_ELEM_IMPL_H
#define LIBTENSOR_BTO_GET_ELEM_IMPL_H

#include "../bto_get_elem.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_get_elem<N, T>::k_clazz = "bto_get_elem<N>";


template<size_t N, typename T>
void bto_get_elem<N, T>::perform(block_tensor_i<N, T> &bt,
    const index<N> &bidx, const index<N> &idx, T& d) {

    m_gbto.perform(bt, bidx, idx, d);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_GET_ELEM_IMPL_H
