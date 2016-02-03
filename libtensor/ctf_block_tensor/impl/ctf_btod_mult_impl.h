#ifndef LIBTENSOR_CTF_BTOD_MULT_IMPL_H
#define LIBTENSOR_CTF_BTOD_MULT_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_mult.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "ctf_btod_set_symmetry.h"
#include "../ctf_btod_mult.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_mult<N>::k_clazz[] = "ctf_btod_mult<N>";


template<size_t N>
void ctf_btod_mult<N>::perform(gen_block_tensor_i<N, bti_traits> &btc) {

    gen_bto_aux_copy<N, ctf_btod_traits> out(get_symmetry(), btc);
    out.open();
    ctf_btod_set_symmetry<N>().perform(get_schedule(), btc);
    perform(out);
    out.close();
}


template<size_t N>
void ctf_btod_mult<N>::perform(gen_block_tensor_i<N, bti_traits> &btc,
    const scalar_transf<double> &d) {

    gen_block_tensor_rd_ctrl<N, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<N, ctf_btod_traits> asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<N, ctf_btod_traits> out(get_symmetry(), asch, btc, d);
    out.open();
    ctf_btod_set_symmetry<N>().perform(asch, btc);
    perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_MULT_IMPL_H
