#ifndef LIBTENSOR_BTOD_DIAG_IMPL_H
#define LIBTENSOR_BTOD_DIAG_IMPL_H

#include <libtensor/block_tensor/bto/bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_diag.h"

namespace libtensor {


template<size_t N, size_t M>
const char *btod_diag<N, M>::k_clazz = "btod_diag<N, M>";


template<size_t N, size_t M>
void btod_diag<N, M>::perform(block_tensor_i<N - M + 1, double> &btb) {

    gen_bto_aux_copy<N - M + 1, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(block_tensor_i<N - M + 1, double> &btb,
        const double &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N - M + 1, bti_traits> cb(btb);
    addition_schedule<N - M + 1, btod_traits> asch(get_symmetry(),
            cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N - M + 1, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(
        dense_tensor_i<N - M + 1, double> &blkb,
        const index<N - M + 1> &ib) {

    m_gbto.compute_block(true, blkb, ib, tensor_transf<N - M + 1, double>());
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(
        bool zero,
        dense_tensor_i<N - M + 1, double> &blkb,
        const index<N - M + 1> &ib,
        const tensor_transf<N - M + 1, double> &trb,
        const double &c) {

    tensor_transf<N - M + 1, double> trx(trb);
    trx.transform(scalar_transf<double>(c));

    m_gbto.compute_block(zero, blkb, ib, trx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_IMPL_H
