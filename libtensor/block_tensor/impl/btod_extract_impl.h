#ifndef LIBTENSOR_BTOD_EXTRACT_IMPL_H
#define LIBTENSOR_BTOD_EXTRACT_IMPL_H

#include <libtensor/block_tensor/bto/bto_aux_add.h>
#include <libtensor/block_tensor/bto/bto_aux_copy.h>
#include "bto_stream_adapter.h"
#include "../btod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char *btod_extract<N, M>::k_clazz = "btod_extract<N, M>";


template<size_t N, size_t M>
void btod_extract<N, M>::perform(bto_stream_i<N - M, btod_traits> &out) {

    bto_stream_adapter<N - M, btod_traits> a(out);
    m_gbto.perform(a);
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(block_tensor_i<N - M, double> &btb) {

    bto_aux_copy<N - M, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(block_tensor_i<N - M, double> &btb,
    const double &c) {

    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<N - M, bti_traits> cb(btb);
    addition_schedule<N - M, btod_traits> asch(get_symmetry(),
            cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N - M, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(
        dense_tensor_i<N - M, double> &blkb, const index<N - M> &idxb) {

    m_gbto.compute_block(true, idxb, tensor_transf_type(), blkb);
}


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(
        bool zero,
        dense_tensor_i<N - M, double> &blkb,
        const index<N - M> &idxb,
        const tensor_transf_type &trb,
        const double &c) {

    tensor_transf_type trx(trb);
    trx.transform(scalar_transf<double>(c));

    m_gbto.compute_block(zero, idxb, trx, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_IMPL_H
