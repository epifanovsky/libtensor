#ifndef LIBTENSOR_CUDA_BTOD_ADD_IMPL_H
#define LIBTENSOR_CUDA_BTOD_ADD_IMPL_H

//#include <libtensor/dense_tensor/tod_copy.h>
//#include <libtensor/dense_tensor/tod_set.h>

#include <libtensor/cuda_dense_tensor/cuda_tod_copy.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../cuda_btod_add.h"

namespace libtensor {


template<size_t N>
const char *cuda_btod_add<N>::k_clazz = "cuda_btod_add<N>";


template<size_t N>
void cuda_btod_add<N>::perform(gen_block_stream_i<N, bti_traits> &out) {

    m_gbto.perform(out);
}


template<size_t N>
void cuda_btod_add<N>::perform(gen_block_tensor_i<N, bti_traits> &btb) {

    gen_bto_aux_copy<N, cuda_btod_traits> out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void cuda_btod_add<N>::perform(gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<double> &c) {

    typedef cuda_block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, cuda_btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    gen_bto_aux_add<N, cuda_btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void cuda_btod_add<N>::perform(cuda_block_tensor_i<N, double> &btb, double c) {

    perform(btb, scalar_transf<double>(c));
}


template<size_t N>
void cuda_btod_add<N>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, double> &trb,
    dense_tensor_wr_i<N, double> &blkb) {

    m_gbto.compute_block(zero, ib, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_ADD_IMPL_H
