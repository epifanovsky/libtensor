#ifndef LIBTENSOR_BTO_CONTRACT3_IMPL_H
#define LIBTENSOR_BTO_CONTRACT3_IMPL_H

#include <libtensor/not_implemented.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract3_impl.h>
#include "../bto_contract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
const char bto_contract3<N1, N2, N3, K1, K2, T>::k_clazz[] =
    "bto_contract3<N1, N2, N3, K1, K2, T>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
bto_contract3<N1, N2, N3, K1, K2, T>::bto_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_rd_i<N1 + K1, T> &bta,
    block_tensor_rd_i<N2 + K1 + K2, T> &btb,
    block_tensor_rd_i<N3 + K2, T> &btc) :

    m_gbto(contr1, contr2, bta, scalar_transf<T>(),
        btb, scalar_transf<T>(), btc, scalar_transf<T>(),
        scalar_transf<T>()) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
bto_contract3<N1, N2, N3, K1, K2, T>::bto_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_rd_i<N1 + K1, T> &bta,
    block_tensor_rd_i<N2 + K1 + K2, T> &btb,
    block_tensor_rd_i<N3 + K2, T> &btc, T kd) :

    m_gbto(contr1, contr2, bta, scalar_transf<T>(),
        btb, scalar_transf<T>(), btc, scalar_transf<T>(),
        scalar_transf<T>(kd)) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_contract3<N1, N2, N3, K1, K2, T>::perform(
    gen_block_stream_i<N1 + N2 + N3, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_contract3<N1, N2, N3, K1, K2, T>::perform(
    gen_block_tensor_i<N1 + N2 + N3, bti_traits> &btd) {

    gen_bto_aux_copy<N1 + N2 + N3, bto_traits<T> > out(m_gbto.get_symmetry(), btd);
    out.open();
    m_gbto.perform(out);
    out.close();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_contract3<N1, N2, N3, K1, K2, T>::perform(
    gen_block_tensor_i<N1 + N2 + N3, bti_traits> &btd,
    const scalar_transf<T> &k) {

    typedef block_tensor_i_traits<T> bti_traits;

    gen_block_tensor_rd_ctrl<N1 + N2 + N3, bti_traits> cd(btd);
    std::vector<size_t> nzblkd;
    cd.req_nonzero_blocks(nzblkd);
    addition_schedule<N1 + N2 + N3, bto_traits<T> > asch(get_symmetry(),
        cd.req_const_symmetry());
    asch.build(get_schedule(), nzblkd);

    gen_bto_aux_add<N1 + N2 + N3, bto_traits<T> > out(get_symmetry(), asch,
        btd, k);
    out.open();
    perform(out);
    out.close();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_contract3<N1, N2, N3, K1, K2, T>::perform(
    block_tensor_i<N1 + N2 + N3, T> &btd,
    T k) {

    perform(btd, scalar_transf<T>(k));
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
void bto_contract3<N1, N2, N3, K1, K2, T>::compute_block(
    bool zero,
    const index<N1 + N2 + N3> &id,
    const tensor_transf<N1 + N2 + N3, T> &trd,
    dense_tensor_wr_i<N1 + N2 + N3, T> &blkd) {

    static const char method[] = "compute_block()";

    throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT3_IMPL_H

