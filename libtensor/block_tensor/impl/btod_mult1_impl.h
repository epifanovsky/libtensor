#ifndef LIBTENSOR_BTOD_MULT1_IMPL_H
#define LIBTENSOR_BTOD_MULT1_IMPL_H

#include "../btod_mult1.h"

namespace libtensor {


template<size_t N>
const char *btod_mult1<N>::k_clazz = "btod_mult1<N>";


template<size_t N>
btod_mult1<N>::btod_mult1(block_tensor_rd_i<N, double> &btb,
        const tensor_transf<N, double> &trb,
        bool recip, const scalar_transf<double> &c) :

    m_gbto(btb, trb, recip, c) {
}


template<size_t N>
btod_mult1<N>::btod_mult1(block_tensor_rd_i<N, double> &btb,
        bool recip, double c) :

    m_gbto(btb, tensor_transf<N, double>(),
            recip, scalar_transf<double>(c)) {
}


template<size_t N>
btod_mult1<N>::btod_mult1(block_tensor_rd_i<N, double> &btb,
        const permutation<N> &pb, bool recip, double c) :

    m_gbto(btb, tensor_transf<N, double>(pb),
            recip, scalar_transf<double>(c)) {
}


template<size_t N>
void btod_mult1<N>::perform(bool zero, block_tensor_i<N, double> &bta) {

    m_gbto.perform(zero, bta);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_IMPL_H
