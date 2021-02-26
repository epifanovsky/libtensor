#ifndef LIBTENSOR_BTO_MULT_IMPL_H
#define LIBTENSOR_BTO_MULT_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../bto_mult.h"

namespace libtensor {


template<size_t N, typename T>
const char bto_mult<N, T>::k_clazz[] = "bto_mult<N, T>";


template<size_t N, typename T>
void bto_mult<N, T>::perform(gen_block_tensor_i<N, bti_traits> &btc) {

    gen_bto_aux_copy<N, bto_traits<T> > out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_mult<N, T>::perform(gen_block_tensor_i<N, bti_traits> &btc,
        const scalar_transf<T> &d) {

    typedef block_tensor_i_traits<T> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<N, bto_traits<T> > asch(get_symmetry(),
            cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<N, bto_traits<T> > out(get_symmetry(), asch, btc, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_mult<N, T>::perform(block_tensor_i<N, T> &btc, T d) {

    perform(btc, scalar_transf<T>(d));
}


template<size_t N, typename T>
void bto_mult<N, T>::compute_block(
        bool zero,
        const index<N> &ic,
        const tensor_transf<N, T> &trc,
        dense_tensor_wr_i<N, T> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_MULT_IMPL_H
