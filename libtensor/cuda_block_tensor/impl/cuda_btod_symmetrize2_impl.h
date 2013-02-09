#ifndef LIBTENSOR_CUDA_BTOD_SYMMETRIZE2_IMPL_H
#define LIBTENSOR_CUDA_BTOD_SYMMETRIZE2_IMPL_H

#include <list>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_symmetrize.h>
#include "../cuda_btod_symmetrize2.h"

namespace libtensor {


template<size_t N>
const char cuda_btod_symmetrize2<N>::k_clazz[] = "cuda_btod_symmetrize2<N>";


template<size_t N>
void cuda_btod_symmetrize2<N>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N>
void cuda_btod_symmetrize2<N>::perform(
    gen_block_tensor_i<N, bti_traits> &bt) {

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<N, double>(get_symmetry()).perform(ctrl.req_symmetry());

    addition_schedule<N, cuda_btod_traits> asch(get_symmetry(), get_symmetry());
    asch.build(get_schedule(), ctrl);

    gen_bto_aux_add<N, cuda_btod_traits> out(get_symmetry(), asch, bt,
        scalar_transf<double>());
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void cuda_btod_symmetrize2<N>::perform(
    gen_block_tensor_i<N, bti_traits> &bt,
    const scalar_transf<double> &d) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    addition_schedule<N, cuda_btod_traits> asch(get_symmetry(),
        ctrl.req_const_symmetry());
    asch.build(get_schedule(), ctrl);

    gen_bto_aux_add<N, cuda_btod_traits> out(get_symmetry(), asch, bt, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void cuda_btod_symmetrize2<N>::perform(
    cuda_block_tensor_i<N, double> &bt,
    double d) {

    perform(bt, scalar_transf<double>(d));
}


template<size_t N>
void cuda_btod_symmetrize2<N>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, double> &trb,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


template<size_t N>
void cuda_btod_symmetrize2<N>::compute_block(
    const index<N> &ib,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(true, ib, tensor_transf<N, double>(), blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_SYMMETRIZE2_IMPL_H
