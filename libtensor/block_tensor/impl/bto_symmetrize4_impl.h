#ifndef LIBTENSOR_BTO_SYMMETRIZE4_IMPL_H
#define LIBTENSOR_BTO_SYMMETRIZE4_IMPL_H

#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../bto_symmetrize4.h"

namespace libtensor {


template<size_t N, typename T>
const char bto_symmetrize4<N, T>::k_clazz[] = "bto_symmetrize4<N, T>";


template<size_t N, typename T>
void bto_symmetrize4<N, T>::perform(gen_block_stream_i<N, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N, typename T>
void bto_symmetrize4<N, T>::perform(gen_block_tensor_i<N, bti_traits> &bt) {

    gen_bto_aux_copy<N, bto_traits<T> > out(get_symmetry(), bt);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_symmetrize4<N, T>::perform(gen_block_tensor_i<N, bti_traits> &bt,
    const scalar_transf<T> &d) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    std::vector<size_t> nzblk;
    ctrl.req_nonzero_blocks(nzblk);
    addition_schedule<N, bto_traits<T> > asch(get_symmetry(),
        ctrl.req_const_symmetry());
    asch.build(get_schedule(), nzblk);

    gen_bto_aux_add<N, bto_traits<T> > out(get_symmetry(), asch, bt, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, typename T>
void bto_symmetrize4<N, T>::perform(block_tensor_i<N, T> &bt, T d) {

    perform(bt, scalar_transf<T>(d));
}


template<size_t N, typename T>
void bto_symmetrize4<N, T>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, T> &trb,
    dense_tensor_wr_i<N, T> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


template<size_t N, typename T>
void bto_symmetrize4<N, T>::compute_block(
    const index<N> &ib,
    dense_tensor_wr_i<N, T> &blkb) {

    m_gbto.compute_block(true, ib, tensor_transf<N, T>(), blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SYMMETRIZE4_IMPL_H
