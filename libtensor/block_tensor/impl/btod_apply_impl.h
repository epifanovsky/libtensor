#ifndef LIBTENSOR_BTOD_APPLY_IMPL_H
#define LIBTENSOR_BTOD_APPLY_IMPL_H

#include <libtensor/block_tensor/bto/bto_aux_add.h>
#include <libtensor/block_tensor/bto/bto_aux_copy.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_apply_impl.h>
#include "bto_stream_adapter.h"
#include "../btod_apply.h"

namespace libtensor {


template<size_t N, typename Functor>
const char *btod_apply<N, Functor>::k_clazz = "btod_apply<N, Functor>";


template<size_t N, typename Functor>
void btod_apply<N, Functor>::perform(bto_stream_i<N, btod_traits> &out) {

    bto_stream_adapter<N, btod_traits> a(out);
    m_gbto.perform(a);
}


template<size_t N, typename Functor>
void btod_apply<N, Functor>::perform(block_tensor_i<N, double> &btb) {

    bto_aux_copy<N, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N, typename Functor>
void btod_apply<N, Functor>::perform(block_tensor_i<N, double> &btb,
    const double &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
            cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N, typename Functor>
void btod_apply<N, Functor>::compute_block(dense_tensor_i<N, double> &blkb,
        const index<N> &ib) {

    m_gbto.compute_block(true, blkb, ib, tensor_transf<N, double>());
}


template<size_t N, typename Functor>
void btod_apply<N, Functor>::compute_block(bool zero,
        dense_tensor_i<N, double> &blkb, const index<N> &ib,
        const tensor_transf<N, double> &trb, const double &c) {

    tensor_transf<N, double> trx(trb);
    trx.transform(scalar_transf<double>(c));

    m_gbto.compute_block(zero, blkb, ib, trb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_APPLY_IMPL_H
