#ifndef LIBTENSOR_BTO_MULT1_IMPL_H
#define LIBTENSOR_BTO_MULT1_IMPL_H

#include "../bto_mult1.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_mult1<N, T>::k_clazz = "bto_mult1<N, T>";


template<size_t N, typename T>
bto_mult1<N, T>::bto_mult1(block_tensor_rd_i<N, T> &btb,
        const tensor_transf<N, T> &trb,
        bool recip, const scalar_transf<T> &c) :

    m_gbto(btb, trb, recip, c) {
}


template<size_t N, typename T>
bto_mult1<N, T>::bto_mult1(block_tensor_rd_i<N, T> &btb,
        bool recip, T c) :

    m_gbto(btb, tensor_transf<N, T>(),
            recip, scalar_transf<T>(c)) {
}


template<size_t N, typename T>
bto_mult1<N, T>::bto_mult1(block_tensor_rd_i<N, T> &btb,
        const permutation<N> &pb, bool recip, T c) :

    m_gbto(btb, tensor_transf<N, T>(pb),
            recip, scalar_transf<T>(c)) {
}


template<size_t N, typename T>
void bto_mult1<N, T>::perform(bool zero, block_tensor_i<N, T> &bta) {

    m_gbto.perform(zero, bta);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_MULT1_IMPL_H
