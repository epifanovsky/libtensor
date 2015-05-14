#ifndef LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_contract2_streamed.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_simple_impl.h>
#include "ctf_btod_set_symmetry.h"
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
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef ctf_dense_tensor_i<NC, double> rd_block_type;
    typedef ctf_dense_tensor_i<NC, double> wr_block_type;

    dimensions<NC> bidimsc = get_bis().get_block_index_dims();

    ctf_block_tensor<NC, double> btc(get_bis());
    gen_block_tensor_ctrl<NC, bti_traits> cc(btc);
    so_copy<NC, double>(get_symmetry()).perform(cc.req_symmetry());

    const assignment_schedule<NC, double> &sch = get_schedule();

    for(typename assignment_schedule<NC, double>::iterator i = sch.begin();
        i != sch.end(); ++i) {

        index<NC> ic;
        abs_index<NC>::get_index(sch.get_abs_index(i), bidimsc, ic);
        tensor_transf<NC, double> trc0;

        std::vector<size_t> blst(1, sch.get_abs_index(i));
        ctf_btod_set_symmetry<NC>().perform(blst, btc);

        {
            wr_block_type &blkc = cc.req_block(ic);
            m_gbto.compute_block(true, ic, trc0, blkc);
            cc.ret_block(ic);
        }
        {
            rd_block_type &blkc = cc.req_const_block(ic);
            out.put(ic, blkc, trc0);
            cc.ret_const_block(ic);
        }
        cc.req_zero_block(ic);
    }
}


template<size_t N, size_t M, size_t K>
void ctf_btod_contract2<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, ctf_btod_traits> out(get_symmetry(), btc);
    out.open();
    ctf_btod_set_symmetry<NC>().perform(get_schedule(), btc);
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
    ctf_btod_set_symmetry<NC>().perform(asch, btc);
    perform(out);
    out.close();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_CONTRACT2_IMPL_H
