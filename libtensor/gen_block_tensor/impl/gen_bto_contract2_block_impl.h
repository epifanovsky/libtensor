#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_IMPL_H

#include "gen_bto_contract2_clst_builder_impl.h"
#include "gen_bto_contract2_block.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_block<N, M, K, Traits, Timed>::gen_bto_contract2_block(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const symmetry<NA, element_type> &syma,
    const block_list<NA> &bla,
    const scalar_transf<element_type> &ka,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NB, element_type> &symb,
    const block_list<NB> &blb,
    const scalar_transf<element_type> &kb,
    const block_index_space<NC> &bisc,
    const scalar_transf<element_type> &kc) :

    m_contr(contr),
    m_bta(bta), m_bta2(bta), m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_syma(syma), m_ola(syma), m_bla(bla), m_ka(ka),
    m_btb(btb), m_btb2(btb), m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_symb(symb), m_olb(symb), m_blb(blb), m_kb(kb),
    m_bidimsc(bisc.get_block_index_dims()), m_kc(kc),
    m_use_broken_sym(false) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_block<N, M, K, Traits, Timed>::gen_bto_contract2_block(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NA, bti_traits> &bta2,
    const symmetry<NA, element_type> &syma,
    const block_list<NA> &bla,
    const scalar_transf<element_type> &ka,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    gen_block_tensor_rd_i<NB, bti_traits> &btb2,
    const symmetry<NB, element_type> &symb,
    const block_list<NB> &blb,
    const scalar_transf<element_type> &kb,
    const block_index_space<NC> &bisc,
    const scalar_transf<element_type> &kc) :

    m_contr(contr),
    m_bta(bta), m_bta2(bta2), m_bidimsa(m_bta2.get_bis().get_block_index_dims()),
    m_syma(syma), m_ola(syma), m_bla(bla), m_ka(ka),
    m_btb(btb), m_btb2(btb2), m_bidimsb(m_btb2.get_bis().get_block_index_dims()),
    m_symb(symb), m_olb(symb), m_blb(blb), m_kb(kb),
    m_bidimsc(bisc.get_block_index_dims()), m_kc(kc),
    m_use_broken_sym(true) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_block<N, M, K, Traits, Timed>::compute_block(
    const contr_list_type &clst,
    bool zero,
    const index<NC> &idxc,
    const tensor_transf<NC, element_type> &trc,
    wr_block_c_type &blkc) {

    typedef typename Traits::template to_contract2_type<N, M, K>::type
        to_contract2;
    typedef typename Traits::template to_set_type<NC>::type to_set;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta), ca2(m_bta2);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb), cb2(m_btb2);

    //  Prepare contraction list
//    gen_bto_contract2_block::start_timer("contract_block::clst");

//    gen_bto_contract2_clst_builder<N, M, K, Traits> clstop(m_contr,
//        m_syma, m_symb, m_bla, m_blb, m_bidimsc, idxc);
//    clstop.build_list(false); // Build full contraction list
//    const contr_list &clst = clstop.get_clst();

//    gen_bto_contract2_block::stop_timer("contract_block::clst");

    //  Keep track of checked out blocks
    typedef std::map<size_t, rd_block_a_type*> coba_map;
    typedef std::map<size_t, rd_block_b_type*> cobb_map;
    coba_map coba;
    cobb_map cobb;

    //  Tensor contraction operation
    std::auto_ptr<to_contract2> op;

    //  Go through the contraction list and prepare the contraction
    for(typename contr_list_type::const_iterator i = clst.begin();
        i != clst.end(); ++i) {

        size_t aia, aib;
        index<NA> ia;
        index<NB> ib;

        if(m_use_broken_sym) {
            aia = i->get_aindex_a();
            aib = i->get_aindex_b();
        } else {
            aia = i->get_acindex_a();
            aib = i->get_acindex_b();
        }
        abs_index<NA>::get_index(aia, m_bidimsa, ia);
        abs_index<NB>::get_index(aib, m_bidimsb, ib);

        if(coba.find(aia) == coba.end()) {
            rd_block_a_type &blka = ca2.req_const_block(ia);
            coba[aia] = &blka;
        }
        if(cobb.find(aib) == cobb.end()) {
            rd_block_b_type &blkb = cb2.req_const_block(ib);
            cobb[aib] = &blkb;
        }
        rd_block_a_type &blka = *coba[aia];
        rd_block_b_type &blkb = *cobb[aib];

        tensor_transf<NA, element_type> tra;
        tensor_transf<NB, element_type> trb;

        if(m_use_broken_sym) {
            orbit<NA, element_type> oa(m_syma, aia);
            orbit<NB, element_type> ob(m_symb, aib);
            tra.transform(tensor_transf<NA, element_type>(
                oa.get_transf(aia), true));
            trb.transform(tensor_transf<NB, element_type>(
                ob.get_transf(aib), true));
        }
        tra.transform(i->get_transf_a());
        trb.transform(i->get_transf_b());

        contraction2<N, M, K> contr(m_contr);
        contr.permute_a(permutation<NA>(tra.get_perm(), true));
        contr.permute_b(permutation<NB>(trb.get_perm(), true));
        contr.permute_c(trc.get_perm());

        scalar_transf<element_type> ka(tra.get_scalar_tr());
        scalar_transf<element_type> kb(trb.get_scalar_tr());
        scalar_transf<element_type> kc(m_kc);

        ka.transform(m_ka);
        kb.transform(m_kb);
        kc.transform(trc.get_scalar_tr());

        if(op.get() == 0) {
            op = std::auto_ptr<to_contract2>(
                new to_contract2(contr, blka, ka, blkb, kb, kc));
        } else {
            op->add_args(contr, blka, ka, blkb, kb, kc);
        }
    }

    //  Execute the contraction
    if(op.get() == 0) {
        if(zero) to_set().perform(blkc);
    } else {
        op->perform(zero, blkc);
    }

    //  Return input blocks
    for(typename coba_map::iterator i = coba.begin(); i != coba.end(); ++i) {
        index<NA> ia;
        abs_index<NA>::get_index(i->first, m_bidimsa, ia);
        ca2.ret_const_block(ia);
    }
    for(typename cobb_map::iterator i = cobb.begin(); i != cobb.end(); ++i) {
        index<NB> ib;
        abs_index<NB>::get_index(i->first, m_bidimsb, ib);
        cb2.ret_const_block(ib);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_IMPL_H
