#ifndef LIBTENSOR_BTOD_MULT_IMPL_H
#define LIBTENSOR_BTOD_MULT_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_mult.h"

namespace libtensor {


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";


template<size_t N>
void btod_mult<N>::perform(gen_block_tensor_i<N, bti_traits> &btc) {

    gen_bto_aux_copy<N, btod_traits> out(get_symmetry(), btc);
    perform(out);
}


template<size_t N>
void btod_mult<N>::perform(gen_block_tensor_i<N, bti_traits> &btc,
        const scalar_transf<double> &d) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cc(btc);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
            cc.req_const_symmetry());
    asch.build(get_schedule(), cc);

    gen_bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btc, d);
    perform(out);
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc, double d) {

    perform(btc, scalar_transf<double>(d));
}


template<size_t N>
void btod_mult<N>::compute_block(
        bool zero,
        const index<N> &ic,
        const tensor_transf<N, double> &trc,
        dense_tensor_i<N, double> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_IMPL_H
