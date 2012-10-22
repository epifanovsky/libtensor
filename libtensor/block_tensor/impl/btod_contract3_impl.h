#ifndef LIBTENSOR_BTOD_CONTRACT3_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT3_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract3_impl.h>
#include "../btod_contract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
const char *btod_contract3<N1, N2, N3, K1, K2>::k_clazz =
    "btod_contract3<N1, N2, N3, K1, K2>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
btod_contract3<N1, N2, N3, K1, K2>::btod_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_i<N1 + K1, double> &bta,
    block_tensor_i<N2 + K1 + K2, double> &btb,
    block_tensor_i<N3 + K2, double> &btc) :

    m_gbto(contr1, contr2, bta, scalar_transf<double>(),
            btb, scalar_transf<double>(), btc, scalar_transf<double>(),
            scalar_transf<double>()) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
btod_contract3<N1, N2, N3, K1, K2>::btod_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_i<N1 + K1, double> &bta,
    block_tensor_i<N2 + K1 + K2, double> &btb,
    block_tensor_i<N3 + K2, double> &btc, double kd) :

    m_gbto(contr1, contr2, bta, scalar_transf<double>(),
            btb, scalar_transf<double>(), btc, scalar_transf<double>(),
            scalar_transf<double>(kd)) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
void btod_contract3<N1, N2, N3, K1, K2>::perform(
        gen_block_stream_i<N1 + N2 + N3, bti_traits> &out) {

    m_gbto.perform(out);
}

template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
void btod_contract3<N1, N2, N3, K1, K2>::perform(
    block_tensor_i<N1 + N2 + N3, double> &btd) {

    gen_bto_aux_copy<N1 + N2 + N3, btod_traits> out(m_gbto.get_symmetry(), btd);
    perform(out);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT3_IMPL_H

