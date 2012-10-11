#ifndef LIBTENSOR_BTOD_RANDOM_IMPL_H
#define LIBTENSOR_BTOD_RANDOM_IMPL_H

#include "../btod_random.h"

namespace libtensor {


template<size_t N>
const char *btod_random<N>::k_clazz = "btod_random<N>";


template<size_t N>
void btod_random<N>::perform(block_tensor_wr_i<N, double> &bt) {

    m_gbto.perform(bt);
}

template<size_t N>
void btod_random<N>::perform(block_tensor_wr_i<N, double> &bt,
        const index<N> &idx) {

    m_gbto.perform(bt, idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_IMPL_H
