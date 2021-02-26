#ifndef LIBTENSOR_BTO_SYMCONTRACT3_IMPL_H
#define LIBTENSOR_BTO_SYMCONTRACT3_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_symcontract3_impl.h>
#include "../bto_symcontract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
const char bto_symcontract3<N1, N2, N3, K1, K2, T>::k_clazz[] =
    "bto_symcontract3<N1, N2, N3, K1, K2>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
bto_symcontract3<N1, N2, N3, K1, K2, T>::bto_symcontract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_rd_i<N1 + K1, T> &bta,
    block_tensor_rd_i<N2 + K1 + K2, T> &btb,
    const permutation<N1 + N2 + K2> &permab,
    bool symmab,
    block_tensor_rd_i<N3 + K2, T> &btc) :

    m_gbto(contr1, contr2, bta, scalar_transf<T>(),
        btb, scalar_transf<T>(), permab, symmab,
        btc, scalar_transf<T>(), scalar_transf<T>()) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
bto_symcontract3<N1, N2, N3, K1, K2, T>::bto_symcontract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_rd_i<N1 + K1, T> &bta,
    block_tensor_rd_i<N2 + K1 + K2, T> &btb,
    const permutation<N1 + N2 + K2> &permab,
    bool symmab,
    block_tensor_rd_i<N3 + K2, T> &btc, T kd) :

    m_gbto(contr1, contr2, bta, scalar_transf<T>(),
        btb, scalar_transf<T>(), permab, symmab,
        btc, scalar_transf<T>(), scalar_transf<T>(kd)) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_symcontract3<N1, N2, N3, K1, K2, T>::perform(
    gen_block_stream_i<N1 + N2 + N3, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_symcontract3<N1, N2, N3, K1, K2, T>::perform(
    block_tensor_i<N1 + N2 + N3, T> &btd) {

    gen_bto_aux_copy<N1 + N2 + N3, bto_traits<T> > out(m_gbto.get_symmetry(),
        btd);
    out.open();
    m_gbto.perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SYMCONTRACT3_IMPL_H

