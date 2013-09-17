#ifndef LIBTENSOR_BTOD_EXTRACT_IMPL_H
#define LIBTENSOR_BTOD_EXTRACT_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../btod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char btod_extract<N, M>::k_clazz[] = "btod_extract<N, M>";


template<size_t N, size_t M>
void btod_extract<N, M>::perform(gen_block_tensor_i<N - M, bti_traits> &btb) {

    gen_bto_aux_copy<N - M, btod_traits> out(get_symmetry(), btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(gen_block_tensor_i<N - M, bti_traits> &btb,
        const scalar_transf<double> &c) {

    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<N - M, bti_traits> cb(btb);
    std::vector<size_t> nzblkb;
    cb.req_nonzero_blocks(nzblkb);
    addition_schedule<N - M, btod_traits> asch(get_symmetry(),
            cb.req_const_symmetry());
    asch.build(get_schedule(), nzblkb);

    gen_bto_aux_add<N - M, btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(block_tensor_i<N - M, double> &btb,
        double c) {

    perform(btb, scalar_transf<double>(c));

}


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(
        bool zero,
        const index<N - M> &idxb,
        const tensor_transf_type &trb,
        dense_tensor_wr_i<N - M, double> &blkb) {

    m_gbto.compute_block(zero, idxb, trb, blkb);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_IMPL_H
