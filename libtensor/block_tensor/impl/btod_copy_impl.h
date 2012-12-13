#ifndef LIBTENSOR_BTOD_COPY_IMPL_H
#define LIBTENSOR_BTOD_COPY_IMPL_H

#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "../btod_copy.h"

namespace libtensor {


template<size_t N>
const char *btod_copy<N>::k_clazz = "btod_copy<N>";


template<size_t N>
void btod_copy<N>::perform(gen_block_tensor_i<N, bti_traits> &btb) {

    gen_bto_aux_copy<N, btod_traits> out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_copy<N>::perform(gen_block_tensor_i<N, bti_traits> &btb,
    const scalar_transf<double> &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    gen_bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb, double c) {

    perform(btb, scalar_transf<double>(c));
}


template<size_t N>
void btod_copy<N>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, double> &trb,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_IMPL_H
