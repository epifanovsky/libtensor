#ifndef LIBTENSOR_CTF_BTOD_EWMULT2_IMPL_H
#define LIBTENSOR_CTF_BTOD_EWMULT2_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_ewmult2.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "ctf_btod_set_symmetry.h"
#include "../ctf_btod_ewmult2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_btod_ewmult2<N, M, K>::k_clazz[] = "ctf_btod_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_btod_ewmult2<N, M, K>::ctf_btod_ewmult2(
    ctf_block_tensor_rd_i<NA, double> &bta,
    ctf_block_tensor_rd_i<NB, double> &btb,
    double d) :

    m_gbto(bta, tensor_transf<NA, double>(), btb, tensor_transf<NB, double>(),
        tensor_transf<NC, double>(permutation<NC>(),
            scalar_transf<double>(d))) {

}


template<size_t N, size_t M, size_t K>
ctf_btod_ewmult2<N, M, K>::ctf_btod_ewmult2(
    ctf_block_tensor_rd_i<NA, double> &bta,
    const permutation<NA> &perma,
    ctf_block_tensor_rd_i<NB, double> &btb,
    const permutation<NB> &permb,
    const permutation<NC> &permc,
    double d) :

    m_gbto(bta, tensor_transf<NA, double>(perma),
        btb, tensor_transf<NB, double>(permb),
        tensor_transf<NC, double>(permc, scalar_transf<double>(d))) {

}


template<size_t N, size_t M, size_t K>
ctf_btod_ewmult2<N, M, K>::ctf_btod_ewmult2(
    ctf_block_tensor_rd_i<NA, double> &bta,
    const tensor_transf<NA, double> &tra,
    ctf_block_tensor_rd_i<NB, double> &btb,
    const tensor_transf<NB, double> &trb,
    const tensor_transf<NC, double> &trc) :

    m_gbto(bta, tra, btb, trb, trc) {

}


template<size_t N, size_t M, size_t K>
void ctf_btod_ewmult2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<N + M + K, ctf_btod_traits> out(get_symmetry(), btc);
    out.open();
    ctf_btod_set_symmetry<NC>().perform(get_schedule(), btc);
    m_gbto.perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void ctf_btod_ewmult2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<double> &d) {

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<NC, ctf_btod_traits> asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<NC, ctf_btod_traits> out(get_symmetry(), asch, btc, d);
    out.open();
    ctf_btod_set_symmetry<NC>().perform(asch, btc);
    m_gbto.perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_IMPL_H
