#ifndef LIBTENSOR_CUDA_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_CUDA_BTOD_CONTRACT2_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_impl.h>
#include "../cuda_btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
cuda_btod_contract2<N, M, K>::cuda_btod_contract2(
    const contraction2<N, M, K> &contr,
    cuda_block_tensor_rd_i<NA, double> &bta,
    cuda_block_tensor_rd_i<NB, double> &btb) :

    m_gbto(contr,
        bta, scalar_transf<double>(),
        btb, scalar_transf<double>(),
        scalar_transf<double>()) {

}


template<size_t N, size_t M, size_t K>
void cuda_btod_contract2<N, M, K>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, size_t M, size_t K>
void cuda_btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, btod_traits> out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void cuda_btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<double> &d) {

    typedef cuda_block_tensor_i_traits<double> bti_traits;

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
void cuda_btod_contract2<N, M, K>::perform(
    cuda_block_tensor_i<NC, double> &btc,
    double d) {

    perform(btc, scalar_transf<double>(d));
}


template<size_t N, size_t M, size_t K>
void cuda_btod_contract2<N, M, K>::compute_block(
    bool zero,
    const index<NC> &ic,
    const tensor_transf<NC, double> &trc,
    dense_tensor_wr_i<NC, double> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_CONTRACT2_IMPL_H
