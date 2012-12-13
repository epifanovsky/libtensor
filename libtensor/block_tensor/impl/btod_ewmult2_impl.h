#ifndef LIBTENSOR_BTOD_EWMULT2_IMPL_H
#define LIBTENSOR_BTOD_EWMULT2_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_ewmult2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_ewmult2<N, M, K>::k_clazz = "btod_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::btod_ewmult2(block_tensor_rd_i<NA, double> &bta,
    block_tensor_rd_i<NB, double> &btb, double d) :

    m_gbto(bta, tensor_transf<NA, double>(), btb, tensor_transf<NB, double>(),
            tensor_transf<NC, double>(permutation<NC>(),
                    scalar_transf<double>(d))) {

}


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::btod_ewmult2(
    block_tensor_rd_i<NA, double> &bta, const permutation<NA> &perma,
    block_tensor_rd_i<NB, double> &btb, const permutation<NB> &permb,
    const permutation<NC> &permc, double d) :

    m_gbto(bta, tensor_transf<NA, double>(perma),
            btb, tensor_transf<NB, double>(permb),
            tensor_transf_type(permc, scalar_transf<double>(d))) {
}


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::btod_ewmult2(
    block_tensor_rd_i<NA, double> &bta, const tensor_transf<NA, double> &tra,
    block_tensor_rd_i<NB, double> &btb, const tensor_transf<NB, double> &trb,
    const tensor_transf_type &trc) :

    m_gbto(bta, tra, btb, trb, trc) {
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::perform(gen_block_stream_i<NC, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::perform(gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<N + M + K, btod_traits> out(get_symmetry(), btc);
    out.open();
    m_gbto.perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::perform(gen_block_tensor_i<NC, bti_traits> &btc,
        const scalar_transf<double> &d) {

    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    addition_schedule<NC, btod_traits> asch(get_symmetry(),
            cc.req_const_symmetry());
    asch.build(get_schedule(), cc);

    gen_bto_aux_add<NC, btod_traits> out(get_symmetry(), asch, btc, d);
    out.open();
    m_gbto.perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::perform(
        block_tensor_i<NC, double> &btc, double d) {

    perform(btc, scalar_transf<double>(d));
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::compute_block(
        bool zero,
        const index<NC> &idx,
        const tensor_transf<NC, double> &tr,
        dense_tensor_wr_i<NC, double> &blk) {

    m_gbto.compute_block(zero, idx, tr, blk);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_IMPL_H
