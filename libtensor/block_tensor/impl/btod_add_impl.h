#ifndef LIBTENSOR_BTOD_ADD_IMPL_H
#define LIBTENSOR_BTOD_ADD_IMPL_H

#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_add.h"

namespace libtensor {


template<size_t N>
const char *btod_add<N>::k_clazz = "btod_add<N>";


template<size_t N>
void btod_add<N>::perform(gen_block_stream_i<N, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &btb) {

    gen_bto_aux_copy<N, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N>
void btod_add<N>::perform(
    block_tensor_i<N, double> &btb,
    const double &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N>
void btod_add<N>::compute_block(
    dense_tensor_i<N, double> &blkb,
    const index<N> &ib) {

    m_gbto.compute_block(true, blkb, ib, tensor_transf<N, double>(), 1.0);
}


template<size_t N>
void btod_add<N>::compute_block(
    bool zero,
    dense_tensor_i<N, double> &blkb,
    const index<N> &ib,
    const tensor_transf<N, double> &trb,
    const double &c) {

    m_gbto.compute_block(zero, blkb, ib, trb, c);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_IMPL_H
