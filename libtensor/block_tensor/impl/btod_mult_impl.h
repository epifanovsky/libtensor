#ifndef LIBTENSOR_BTOD_MULT_IMPL_H
#define LIBTENSOR_BTOD_MULT_IMPL_H

#include <libtensor/block_tensor/bto/bto_aux_add.h>
#include <libtensor/block_tensor/bto/bto_aux_copy.h>
#include "bto_stream_adapter.h"
#include "../btod_mult.h"

namespace libtensor {


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";



template<size_t N>
void btod_mult<N>::perform(bto_stream_i<N, btod_traits> &out) {

    bto_stream_adapter<N, btod_traits> a(out);
    m_gbto.perform(a);
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc) {

    bto_aux_copy<N, btod_traits> out(get_symmetry(), btc);
    perform(out);
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc, const double &d) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cc(btc);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
            cc.req_const_symmetry());
    asch.build(get_schedule(), cc);

    bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btc, d);
    perform(out);
}


template<size_t N>
void btod_mult<N>::compute_block(dense_tensor_i<N, double> &blkc,
        const index<N> &ic) {

    m_gbto.compute_block(true, blkc, ic, tensor_transf<N, double>());
}

template<size_t N>
void btod_mult<N>::compute_block(bool zero, dense_tensor_i<N, double> &blkc,
    const index<N> &ic, const tensor_transf<N, double> &trc,
    const double &c) {

    tensor_transf<N, double> trx(trc);
    trx.transform(scalar_transf<double>(c));

    m_gbto.compute_block(zero, blkc, ic, trx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_IMPL_H
