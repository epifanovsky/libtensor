#ifndef LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_contract2_streamed.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_simple_impl.h>
#include "../ctf_btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_btod_contract2<N, M, K>::k_clazz[] =
    "ctf_btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_btod_contract2<N, M, K>::ctf_btod_contract2(
    const contraction2<N, M, K> &contr,
    ctf_block_tensor_rd_i<NA, double> &bta,
    ctf_block_tensor_rd_i<NB, double> &btb) :

    m_gbto(contr,
        bta, scalar_transf<double>(),
        btb, scalar_transf<double>(),
        scalar_transf<double>()) {

}


template<size_t N, size_t M, size_t K>
ctf_btod_contract2<N, M, K>::ctf_btod_contract2(
    const contraction2<N, M, K> &contr,
    ctf_block_tensor_rd_i<NA, double> &bta,
    double ka,
    ctf_block_tensor_rd_i<NB, double> &btb,
    double kb,
    double kc) :

    m_gbto(contr,
        bta, scalar_transf<double>(ka),
        btb, scalar_transf<double>(kb),
        scalar_transf<double>(kc)) {

}


template<size_t N, size_t M, size_t K>
void ctf_btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, ctf_btod_traits> out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void ctf_btod_contract2<N, M, K>::perform(
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
    perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H
