#ifndef LIBTENSOR_BTOD_DIAG_IMPL_H
#define LIBTENSOR_BTOD_DIAG_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_diag.h"

namespace libtensor {


template<size_t N, size_t M>
const char btod_diag<N, M>::k_clazz[] = "btod_diag<N, M>";


template<size_t N, size_t M>
void btod_diag<N, M>::perform(gen_block_tensor_i<M, bti_traits> &btb) {

    gen_bto_aux_copy<M, btod_traits> out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(gen_block_tensor_i<M, bti_traits> &btb,
        const scalar_transf<double> &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<M, bti_traits> cb(btb);
    std::vector<size_t> nzblkb;
    cb.req_nonzero_blocks(nzblkb);
    addition_schedule<M, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), nzblkb);

    gen_bto_aux_add<M, btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(block_tensor_i<M, double> &btb, double c) {

    perform(btb, scalar_transf<double>(c));
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(
        bool zero,
        const index<M> &ib,
        const tensor_transf<M, double> &trb,
        dense_tensor_wr_i<M, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_IMPL_H
