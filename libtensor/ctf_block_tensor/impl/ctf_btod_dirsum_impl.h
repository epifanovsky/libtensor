#ifndef LIBTENSOR_CTF_BTOD_DIRSUM_IMPL_H
#define LIBTENSOR_CTF_BTOD_DIRSUM_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_dirsum.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_scatter.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "ctf_btod_set_symmetry.h"
#include "../ctf_btod_dirsum.h"

namespace libtensor {


template<size_t N, size_t M>
const char ctf_btod_dirsum<N, M>::k_clazz[] = "ctf_btod_dirsum<N, M>";


template<size_t N, size_t M>
void ctf_btod_dirsum<N, M>::perform(
    gen_block_tensor_i<N + M, bti_traits> &btb) {

    gen_bto_aux_copy<N + M, ctf_btod_traits> out(get_symmetry(), btb);
    out.open();
    ctf_btod_set_symmetry<N + M>().perform(get_schedule(), btb);
    perform(out);
    out.close();
}


template<size_t N, size_t M>
void ctf_btod_dirsum<N, M>::perform(
    gen_block_tensor_i<N + M, bti_traits> &btb,
    const scalar_transf<double> &c) {

    gen_block_tensor_rd_ctrl<N + M, bti_traits> cb(btb);
    std::vector<size_t> nzblkb;
    cb.req_nonzero_blocks(nzblkb);
    addition_schedule<N + M, ctf_btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), nzblkb);

    gen_bto_aux_add<N + M, ctf_btod_traits> out(get_symmetry(), asch, btb, c);
    out.open();
    ctf_btod_set_symmetry<N + M>().perform(asch, btb);
    perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENOSR_CTF_BTOD_DIRSUM_IMPL_H
