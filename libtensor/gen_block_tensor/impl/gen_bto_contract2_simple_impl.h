#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_SIMPLE_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_SIMPLE_IMPL_H

#include <iterator>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_contract2_align.h"
#include "gen_bto_contract2_batch_impl.h"
#include "gen_bto_contract2_batching_policy.h"
#include "gen_bto_contract2_clst_builder.h"
#include "gen_bto_contract2_nzorb_impl.h"
#include "gen_bto_contract2_sym_impl.h"
#include "gen_bto_unfold_block_list.h"
#include "gen_bto_unfold_symmetry.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_transform.h"
#include "../gen_bto_contract2_simple.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_simple<N, M, K, Traits, Timed>::gen_bto_contract2_simple(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const scalar_transf<element_type> &ka,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const scalar_transf<element_type> &kb,
    const scalar_transf<element_type> &kc) :

    m_contr(contr), m_bta(bta), m_ka(ka), m_btb(btb), m_kb(kb),
    m_kc(kc), m_symc(contr, bta, btb),
    m_sch(m_symc.get_bis().get_block_index_dims()) {

    make_schedule();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_simple<N, M, K, Traits, Timed>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_c_type;

    gen_bto_contract2_simple::start_timer();

    try {

    dimensions<NC> bidimsc = m_symc.get_bis().get_block_index_dims();

    temp_block_tensor_c_type btc(m_symc.get_bis());
    gen_block_tensor_ctrl<NC, bti_traits> cc(btc);

    for(typename assignment_schedule<NC, element_type>::iterator i =
        m_sch.begin(); i != m_sch.end(); ++i) {

        index<NC> ic;
        abs_index<NC>::get_index(m_sch.get_abs_index(i), bidimsc, ic);
        tensor_transf<NC, element_type> trc0;
        {
            wr_block_type &blkc = cc.req_block(ic);
            compute_block(true, ic, trc0, blkc);
            cc.ret_block(ic);
        }
        {
            rd_block_type &blkc = cc.req_const_block(ic);
            out.put(ic, blkc, trc0);
            cc.ret_const_block(ic);
        }
        cc.req_zero_block(ic);
    }

    } catch(...) {
        gen_bto_contract2_simple::stop_timer();
        throw;
    }

    gen_bto_contract2_simple::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_simple<N, M, K, Traits, Timed>::compute_block(
    bool zero,
    const index<NC> &idxc,
    const tensor_transf<NC, double> &trc,
    wr_block_type &blkc) {

    dimensions<NA> bidimsa = m_bta.get_bis().get_block_index_dims();
    dimensions<NB> bidimsb = m_btb.get_bis().get_block_index_dims();
    dimensions<NC> bidimsc = m_symc.get_bis().get_block_index_dims();

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

    std::vector<size_t> blsta, blstb;
    ca.req_nonzero_blocks(blsta);
    cb.req_nonzero_blocks(blstb);
    block_list<NA> bla(bidimsa, blsta), blax(bidimsa);
    block_list<NB> blb(bidimsb, blstb), blbx(bidimsb);

    const symmetry<NA, element_type> &syma = ca.req_const_symmetry();
    const symmetry<NB, element_type> &symb = cb.req_const_symmetry();

    gen_bto_unfold_block_list<NA, Traits>(syma, bla).build(blax);
    gen_bto_unfold_block_list<NB, Traits>(symb, blb).build(blbx);

    gen_bto_contract2_block<N, M, K, Traits, Timed> bto(m_contr, m_bta,
        syma, bla, m_ka, m_btb, symb, blb, m_kb, m_symc.get_bis(), m_kc);

    gen_bto_contract2_clst_builder<N, M, K, Traits> clstop(m_contr,
        syma, symb, blax, blbx, bidimsc, idxc);
    clstop.build_list(false); // Build full contraction list

    bto.compute_block(clstop.get_clst(), zero, idxc, trc, blkc);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_simple<N, M, K, Traits, Timed>::make_schedule() {

    gen_bto_contract2_simple::start_timer("make_schedule");

    gen_bto_contract2_nzorb<N, M, K, Traits> nzorb(m_contr, m_bta, m_btb,
        m_symc.get_symmetry());

    nzorb.build();
    const block_list<NC> &blstc = nzorb.get_blst();
    for(typename block_list<NC>::iterator i = blstc.begin();
            i != blstc.end(); ++i) {
        m_sch.insert(blstc.get_abs_index(i));
    }

    gen_bto_contract2_simple::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_SIMPLE_IMPL_H
