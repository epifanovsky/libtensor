#ifndef LIBTENSOR_BTOD_DIRSUM_IMPL_H
#define LIBTENSOR_BTOD_DIRSUM_IMPL_H

#include <libtensor/block_tensor/bto/bto_aux_add.h>
#include <libtensor/block_tensor/bto/bto_aux_copy.h>
#include "bto_stream_adapter.h"
#include "../btod_dirsum.h"

namespace libtensor {


template<size_t N, size_t M>
const char *btod_dirsum_clazz<N, M>::k_clazz = "btod_dirsum<N, M>";


template<size_t N, size_t M>
const char *btod_dirsum<N, M>::k_clazz = btod_dirsum_clazz<N, M>::k_clazz;


template<size_t N, size_t M>
void btod_dirsum<N, M>::perform(bto_stream_i<N + M, btod_traits> &out) {

    bto_stream_adapter<N + M, btod_traits> a(out);
    m_gbto.perform(a);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::perform(block_tensor_i<N + M, double> &btb) {

    bto_aux_copy<N + M, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::perform(block_tensor_i<N + M, double> &btb,
    const double &c) {

    block_tensor_ctrl<N + M, double> cb(btb);
    addition_schedule<N + M, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N + M, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::compute_block(
        dense_tensor_i<N + M, double> &blkc,
        const index<N + M> &ic) {

    m_gbto.compute_block(true, ic, tensor_transf<N + M, double>(), blkc);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::compute_block(
        bool zero,
        dense_tensor_i<N + M, double> &blkc,
        const index<N + M> &ic,
        const tensor_transf<N + M, double> &trc,
        const double &c) {

    tensor_transf<N + M, double> trx(trc);
    trx.transform(scalar_transf<double>(c));

    m_gbto.compute_block(zero, ic, trx, blkc);
}


} // namespace libtensor


#endif // LIBTENOSR_BTOD_DIRSUM_IMPL_H
