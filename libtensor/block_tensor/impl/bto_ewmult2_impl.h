#ifndef LIBTENSOR_BTO_EWMULT2_IMPL_H
#define LIBTENSOR_BTO_EWMULT2_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../bto_ewmult2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
const char bto_ewmult2<N, M, K, T>::k_clazz[] = "bto_ewmult2<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
bto_ewmult2<N, M, K, T>::bto_ewmult2(block_tensor_rd_i<NA, T> &bta,
    block_tensor_rd_i<NB, T> &btb, T d) :

    m_gbto(bta, tensor_transf<NA, T>(), btb, tensor_transf<NB, T>(),
            tensor_transf<NC, T>(permutation<NC>(),
                    scalar_transf<T>(d))) {

}


template<size_t N, size_t M, size_t K, typename T>
bto_ewmult2<N, M, K, T>::bto_ewmult2(
    block_tensor_rd_i<NA, T> &bta, const permutation<NA> &perma,
    block_tensor_rd_i<NB, T> &btb, const permutation<NB> &permb,
    const permutation<NC> &permc, T d) :

    m_gbto(bta, tensor_transf<NA, T>(perma),
            btb, tensor_transf<NB, T>(permb),
            tensor_transf_type(permc, scalar_transf<T>(d))) {
}


template<size_t N, size_t M, size_t K, typename T>
bto_ewmult2<N, M, K, T>::bto_ewmult2(
    block_tensor_rd_i<NA, T> &bta, const tensor_transf<NA, T> &tra,
    block_tensor_rd_i<NB, T> &btb, const tensor_transf<NB, T> &trb,
    const tensor_transf_type &trc) :

    m_gbto(bta, tra, btb, trb, trc) {
}


template<size_t N, size_t M, size_t K, typename T>
void bto_ewmult2<N, M, K, T>::perform(gen_block_stream_i<NC, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, size_t M, size_t K, typename T>
void bto_ewmult2<N, M, K, T>::perform(gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<N + M + K, bto_traits<T> > out(get_symmetry(), btc);
    out.open();
    m_gbto.perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K, typename T>
void bto_ewmult2<N, M, K, T>::perform(gen_block_tensor_i<NC, bti_traits> &btc,
        const scalar_transf<T> &d) {

    typedef typename bto_traits<T> ::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<NC, bto_traits<T> > asch(get_symmetry(),
            cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<NC, bto_traits<T> > out(get_symmetry(), asch, btc, d);
    out.open();
    m_gbto.perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K, typename T>
void bto_ewmult2<N, M, K, T>::perform(
        block_tensor_i<NC, T> &btc, T d) {

    perform(btc, scalar_transf<T>(d));
}


template<size_t N, size_t M, size_t K, typename T>
void bto_ewmult2<N, M, K, T>::compute_block(
        bool zero,
        const index<NC> &idx,
        const tensor_transf<NC, T> &tr,
        dense_tensor_wr_i<NC, T> &blk) {

    m_gbto.compute_block(zero, idx, tr, blk);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_EWMULT2_IMPL_H
