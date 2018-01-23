#ifndef LIBTENSOR_BTO_RANDOM_IMPL_H
#define LIBTENSOR_BTO_RANDOM_IMPL_H

#include "../bto_random.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_random<N, T>::k_clazz = "bto_random<N, T>";


template<size_t N, typename T>
void bto_random<N, T>::perform(block_tensor_wr_i<N, T> &bt) {

    m_gbto.perform(bt);
}

template<size_t N, typename T>
void bto_random<N, T>::perform(block_tensor_wr_i<N, T> &bt,
        const index<N> &idx) {

    m_gbto.perform(bt, idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_RANDOM_IMPL_H
