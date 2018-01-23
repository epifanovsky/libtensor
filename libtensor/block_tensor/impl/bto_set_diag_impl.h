#ifndef LIBTENSOR_BTO_SET_DIAG_IMPL_H
#define LIBTENSOR_BTO_SET_DIAG_IMPL_H

#include "../bto_set_diag.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_set_diag<N, T>::k_clazz = "bto_set_diag<N, T>";


template<size_t N, typename T>
bto_set_diag<N, T>::bto_set_diag(const sequence<N, size_t> &msk, T v) :
    m_gbto(msk, v) {

}


template<size_t N, typename T>
bto_set_diag<N, T>::bto_set_diag(T v) : m_gbto(sequence<N, size_t>(1), v) {

}


template<size_t N, typename T>
void bto_set_diag<N, T>::perform(block_tensor_i<N, T> &bt) {

    m_gbto.perform(bt);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_DIAG_IMPL_H
