#ifndef LIBTENSOR_CTF_BTOD_SYMMETRIZE2_IMPL_H
#define LIBTENSOR_CTF_BTOD_SYMMETRIZE2_IMPL_H

#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include "../ctf_btod_symmetrize2.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_symmetrize2<N>::k_clazz[] = "ctf_btod_symmetrize2<N>";


template<size_t N>
void ctf_btod_symmetrize2<N>::perform(
    gen_block_tensor_i<N, bti_traits> &bt) {

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<N, double>(get_symmetry()).perform(ctrl.req_symmetry());

    std::vector<size_t> nzblk;
    addition_schedule<N, ctf_btod_traits> asch(get_symmetry(), get_symmetry());
    asch.build(get_schedule(), nzblk);

    gen_bto_aux_add<N, ctf_btod_traits> out(get_symmetry(), asch, bt,
        scalar_transf<double>());
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void ctf_btod_symmetrize2<N>::perform(
    gen_block_tensor_i<N, bti_traits> &bt,
    const scalar_transf<double> &d) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    std::vector<size_t> nzblk;
    ctrl.req_nonzero_blocks(nzblk);
    addition_schedule<N, ctf_btod_traits> asch(get_symmetry(),
        ctrl.req_const_symmetry());
    asch.build(get_schedule(), nzblk);

    gen_bto_aux_add<N, ctf_btod_traits> out(get_symmetry(), asch, bt, d);
    out.open();
    perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SYMMETRIZE2_IMPL_H
