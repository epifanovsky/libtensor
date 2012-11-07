#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2.h>
#include "../btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2_clazz<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz =
    btod_contract2_clazz<N, M, K>::k_clazz;


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, double> &bta,
    block_tensor_rd_i<NB, double> &btb) :

    m_gbto(contr,
        bta, scalar_transf<double>(),
        btb, scalar_transf<double>(),
        scalar_transf<double>()) {

}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, btod_traits> out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<double> &d) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    addition_schedule<NC, btod_traits> asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), cc);

    gen_bto_aux_add<NC, btod_traits> out(get_symmetry(), asch, btc, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(
    block_tensor_i<NC, double> &btc,
    double d) {

    perform(btc, scalar_transf<double>(d));
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(
    bool zero,
    const index<NC> &ic,
    const tensor_transf<NC, double> &trc,
    dense_tensor_wr_i<NC, double> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
