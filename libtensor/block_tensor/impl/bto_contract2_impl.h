#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_impl.h>
#include "../bto_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char bto_contract2_clazz<N, M, K>::k_clazz[] = "bto_contract2<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
const char bto_contract2<N, M, K, T>::k_clazz[] = "bto_contract2<N, M, K>";


template<size_t N, size_t M, size_t K, typename T>
bto_contract2<N, M, K, T>::bto_contract2(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, T> &bta,
    block_tensor_rd_i<NB, T> &btb) :

    m_gbto(contr,
        bta, scalar_transf<T>(),
        btb, scalar_transf<T>(),
        scalar_transf<T>()) {

}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2<N, M, K, T>::bto_contract2(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, T> &bta,
    T ka,
    block_tensor_rd_i<NB, T> &btb,
    T kb,
    T kc) :

    m_gbto(contr,
        bta, scalar_transf<T>(ka),
        btb, scalar_transf<T>(kb),
        scalar_transf<T>(kc)) {

}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2<N, M, K, T>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2<N, M, K, T>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, bto_traits<T> > out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2<N, M, K, T>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<T> &d) {

    typedef block_tensor_i_traits<T> bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<NC, bto_traits<T> > asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<NC, bto_traits<T> > out(get_symmetry(), asch, btc, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2<N, M, K, T>::perform(
    block_tensor_i<NC, T> &btc,
    T d) {

    perform(btc, scalar_transf<T>(d));
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2<N, M, K, T>::compute_block(
    bool zero,
    const index<NC> &ic,
    const tensor_transf<NC, T> &trc,
    dense_tensor_wr_i<NC, T> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
