#ifndef LIBTENSOR_BTO_SET_ELEM_IMPL_H
#define LIBTENSOR_BTO_SET_ELEM_IMPL_H

#include "../bto_set_elem.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_set_elem<N, T>::k_clazz = "bto_set_elem<N>";


template<size_t N, typename T>
void bto_set_elem<N, T>::perform(block_tensor_i<N, T> &bt,
    const index<N> &bidx, const index<N> &idx, T d) {

    m_gbto.perform(bt, bidx, idx, d);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_ELEM_IMPL_H
