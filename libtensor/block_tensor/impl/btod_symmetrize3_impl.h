#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H

#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_symmetrize3.h"

namespace libtensor {


template<size_t N>
const char btod_symmetrize3<N>::k_clazz[] = "btod_symmetrize3<N>";


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_stream_i<N, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_tensor_i<N, bti_traits> &bt) {

    gen_bto_aux_copy<N, btod_traits> out(get_symmetry(), bt);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_tensor_i<N, bti_traits> &bt,
    const scalar_transf<double> &d) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    addition_schedule<N, btod_traits> asch(get_symmetry(),
        ctrl.req_const_symmetry());
    asch.build(get_schedule(), ctrl);

    gen_bto_aux_add<N, btod_traits> out(get_symmetry(), asch, bt, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_symmetrize3<N>::perform(block_tensor_i<N, double> &bt, double d) {

    perform(bt, scalar_transf<double>(d));
}


template<size_t N>
void btod_symmetrize3<N>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, double> &trb,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


template<size_t N>
void btod_symmetrize3<N>::compute_block(
    const index<N> &ib,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(true, ib, tensor_transf<N, double>(), blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H
