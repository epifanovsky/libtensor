#ifndef LIBTENSOR_TOD_COPY_IMPL_H
#define LIBTENSOR_TOD_COPY_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "../bto_copy.h"

namespace libtensor {


template<size_t N, typename T>
const char bto_copy<N, T>::k_clazz[] = "bto_copy<N, T>";


template<size_t N, typename T>
void bto_copy<N, T>::perform(gen_block_tensor_i<N, bti_traits> &btb) {

    gen_bto_aux_copy<N, bto_traits<T> > out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_copy<N, T>::perform(gen_block_tensor_i<N, bti_traits> &btb,
    const scalar_transf<T> &c) {

    typedef block_tensor_i_traits<T> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    std::vector<size_t> nzblkb;
    cb.req_nonzero_blocks(nzblkb);
    addition_schedule<N, bto_traits<T> > asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), nzblkb);

    gen_bto_aux_add<N, bto_traits<T> > out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_copy<N, T>::perform(block_tensor_i<N, T> &btb, T c) {

    perform(btb, scalar_transf<T>(c));
}


template<size_t N, typename T>
void bto_copy<N, T>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, T> &trb,
    dense_tensor_wr_i<N, T> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_IMPL_H
