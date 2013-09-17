#ifndef LIBTENSOR_BTOD_DIAG_IMPL_H
#define LIBTENSOR_BTOD_DIAG_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_diag.h"

namespace libtensor {


template<size_t N, size_t M>
const char btod_diag<N, M>::k_clazz[] = "btod_diag<N, M>";


template<size_t N, size_t M>
void btod_diag<N, M>::perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb) {

    gen_bto_aux_copy<N - M + 1, btod_traits> out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb,
        const scalar_transf<double> &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N - M + 1, bti_traits> cb(btb);
    std::vector<size_t> nzblkb;
    cb.req_nonzero_blocks(nzblkb);
    addition_schedule<N - M + 1, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), nzblkb);

    gen_bto_aux_add<N - M + 1, btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(
        block_tensor_i<N - M + 1, double> &btb, double c) {

    perform(btb, scalar_transf<double>(c));
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(
        bool zero,
        const index<N - M + 1> &ib,
        const tensor_transf<N - M + 1, double> &trb,
        dense_tensor_wr_i<N - M + 1, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_IMPL_H
