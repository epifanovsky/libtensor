#ifndef LIBTENSOR_GEN_BTO_SYMCONTRACT3_IMPL_H
#define LIBTENSOR_GEN_BTO_SYMCONTRACT3_IMPL_H

#include <iterator>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_copy_impl.h"
#include "gen_bto_contract2_align.h"
#include "gen_bto_contract2_batch_impl.h"
#include "gen_bto_contract2_clst_builder.h"
#include "gen_bto_contract2_nzorb.h"
#include "gen_bto_contract2_sym_impl.h"
#include "gen_bto_contract3_batching_policy.h"
#include "gen_bto_symcontract2_sym_impl.h"
#include "gen_bto_set_impl.h"
#include "gen_bto_unfold_symmetry.h"
#include "../addition_schedule.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_copy.h"
#include "../gen_bto_aux_symmetrize.h"
#include "../gen_bto_aux_transform.h"
#include "../gen_bto_symcontract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
gen_bto_symcontract3<N1, N2, N3, K1, K2, Traits, Timed>::gen_bto_symcontract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const scalar_transf<element_type> &ka,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const scalar_transf<element_type> &kb,
    const permutation<NAB> &sympermab,
    bool symmab,
    gen_block_tensor_rd_i<NC, bti_traits> &btc,
    const scalar_transf<element_type> &kc,
    const scalar_transf<element_type> &kd) :

    m_contr1(contr1), m_contr2(contr2), m_bta(bta), m_ka(ka),
    m_btb(btb), m_kb(kb), m_sympermab(sympermab), m_symmab(symmab),
    m_btc(btc), m_kc(kc), m_kd(kd),
    m_symab(contr1, bta, btb, sympermab, symmab),
    m_symd(contr2, m_symab.get_symmetry(), retrieve_symmetry(btc)),
    m_schab(m_symab.get_bis().get_block_index_dims()),
    m_schd(m_symd.get_bis().get_block_index_dims()) {

    make_schedule();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_symcontract3<N1, N2, N3, K1, K2, Traits, Timed>::perform(
    gen_block_stream_i<ND, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
            temp_block_tensor_c_type;
    typedef typename Traits::template temp_block_tensor_type<ND>::type
            temp_block_tensor_d_type;
    typedef typename Traits::template temp_block_tensor_type<NAB>::type
            temp_block_tensor_ab_type;

    typedef gen_bto_set<NC, Traits, Timed> gen_bto_set_c_type;
    typedef gen_bto_copy<NAB, Traits, Timed> gen_bto_copy_ab_type;
    typedef gen_bto_copy<NC, Traits, Timed> gen_bto_copy_c_type;

    gen_bto_symcontract3::start_timer();

    try {

        //  Compute the number of blocks in A, B, C, and D

        std::vector<size_t> blsta, blstb, blstc;

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            gen_block_tensor_rd_ctrl<NC, bti_traits> cc(m_btc);
            ca.req_nonzero_blocks(blsta);
            cb.req_nonzero_blocks(blstb);
            cc.req_nonzero_blocks(blstc);
        }

        size_t nblka = blsta.size(), nblkb = blstb.size(),
            nblkc = blstc.size(), nblkab = 0, nblkd = 0;
        nblkab = std::distance(m_schab.begin(), m_schab.end());
        nblkd = std::distance(m_schd.begin(), m_schd.end());

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0 || nblkc == 0 || nblkab == 0) {
            gen_bto_symcontract3::stop_timer();
            return;
        }

        //  Compute optimal permutations of A and B to perform 1st contraction

        gen_bto_contract2_align<N1, N2 + K2, K1> align1(m_contr1);

        const permutation<NA> &perma = align1.get_perma();
        const permutation<NB> &permb = align1.get_permb();
        const permutation<NAB> &permab1 = align1.get_permc();
        permutation<NAB> sympermab(permab1, true);
        sympermab.permute(m_sympermab);
        sympermab.permute(permab1);

        // Prepare permuted arguments of 1st contraction

        contraction2<N1, N2 + K2, K1> contr1(m_contr1);
        contr1.permute_a(perma);
        contr1.permute_b(permb);
        contr1.permute_c(permab1);

        block_index_space<NA> bisat(m_bta.get_bis());
        bisat.permute(perma);
        block_index_space<NB> bisbt(m_btb.get_bis());
        bisbt.permute(permb);
        block_index_space<NAB> bisab1(m_symab.get_bis());
        bisab1.permute(permab1);

        symmetry<NA, element_type> symat(bisat);
        symmetry<NB, element_type> symbt(bisbt);
        symmetry<NAB, element_type> symab10(bisab1), symab1(bisab1);
        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            so_permute<NA, element_type>(ca.req_const_symmetry(), perma).
                perform(symat);
            so_permute<NB, element_type>(cb.req_const_symmetry(), permb).
                perform(symbt);
            so_permute<NAB, element_type>(m_symab.get_symmetry0(), permab1).
                perform(symab10);
            so_permute<NAB, element_type>(m_symab.get_symmetry(), permab1).
                perform(symab1);
        }

        // Compute optimal permutations of AB and C to perform 2nd contraction

        gen_bto_contract2_align<N1 + N2, N3, K2> align2(m_contr2);

        const permutation<NAB> &permab2 = align2.get_perma();
        const permutation<NC> &permc = align2.get_permb();
        const permutation<ND> &permd = align2.get_permc();
        permutation<NAB> permab(permab1, true);
        permab.permute(permab2);
        permutation<ND> permdinv(permd, true);

        // Prepare permuted arguments of 2nd contraction

        contraction2<N1 + N2, N3, K2> contr2(m_contr2);
        contr2.permute_a(permab2);
        contr2.permute_b(permc);
        contr2.permute_c(permd);

        block_index_space<NAB> bisab2(m_symab.get_bis());
        bisab2.permute(permab2);
        block_index_space<NC> bisct(m_btc.get_bis());
        bisct.permute(permc);
        block_index_space<ND> bisdt(m_symd.get_bis());
        bisdt.permute(permd);

        symmetry<NAB, element_type> symab2(bisab2);
        symmetry<NC, element_type> symct(bisct);
        symmetry<ND, element_type> symdt(bisdt);
        {
            gen_block_tensor_rd_ctrl<NC, bti_traits> cc(m_btc);
            so_permute<NAB, element_type>(m_symab.get_symmetry(), permab2).
                perform(symab2);
            so_permute<NC, element_type>(cc.req_const_symmetry(), permc).
                perform(symct);
            so_permute<ND, element_type>(m_symd.get_symmetry(), permd).
                perform(symdt);
        }

        //  Temporary partial AB1, AB2, C, and D

        temp_block_tensor_ab_type btab1(bisab1), btab2(bisab2);
        temp_block_tensor_d_type btdt(bisdt);

        gen_block_tensor_rd_ctrl<ND, bti_traits> cdt(btdt);

        //  Batching loops

        dimensions<NA> bidimsa(m_bta.get_bis().get_block_index_dims());
        dimensions<NB> bidimsb(m_btb.get_bis().get_block_index_dims());
        dimensions<NC> bidimsc(m_btc.get_bis().get_block_index_dims());
        dimensions<NAB> bidimsab(m_symab.get_bis().get_block_index_dims());
        dimensions<NAB> bidimsab1(bisab1.get_block_index_dims());
        dimensions<NAB> bidimsab2(bisab2.get_block_index_dims());
        dimensions<ND> bidimsd(m_symd.get_bis().get_block_index_dims());
        dimensions<ND> bidimsdt(bisdt.get_block_index_dims());

        scalar_transf<element_type> kab;

        gen_bto_contract3_batching_policy<N1, N2, N3, K1, K2> bp(m_contr1,
            m_contr2, nblka, nblkb, nblkc, nblkab, nblkd);
        size_t batchsza = bp.get_bsz_a(), batchszb = bp.get_bsz_b(),
            batchszab = bp.get_bsz_ab(), batchszc = bp.get_bsz_c(),
            batchszd = bp.get_bsz_d();

        std::vector<size_t> batchab1, batchab2, batchc, batchd;
        batchab1.reserve(batchszab);
        batchab2.reserve(batchszab);
        batchc.reserve(batchszc);
        batchd.reserve(batchszd);

        block_index_space<NC> bisc2(m_btc.get_bis());
        bisc2.permute(permc);
        dimensions<NC> bidimsc2 = bisc2.get_block_index_dims();
        temp_block_tensor_c_type btc2(bisc2);
        symmetry<NC, element_type> symc2(bisc2);
        {
            gen_block_tensor_rd_ctrl<NC, bti_traits> cc(m_btc);
            so_permute<NC, element_type>(cc.req_const_symmetry(), permc).
                perform(symc2);
        }
        std::vector<size_t> blstc2;

        //  Use addition schedule as a proxy to obtain common symmetry
        //  subgroup and corresponding schedule for original and
        //  symmetrized intermediate AB
        addition_schedule<NAB, Traits> addschab(m_symab.get_symmetry0(),
            m_symab.get_symmetry());
        addschab.build(m_schab, std::vector<size_t>());

        typename addition_schedule<NAB, Traits>::iterator ibab =
            addschab.begin();
        while(ibab != addschab.end()) {

            batchab1.clear();
            batchab2.clear();
            typedef typename addition_schedule<NAB, Traits>::schedule_group
                schedule_group;
            if(permab1.is_identity()) {
                for(; ibab != addschab.end() && batchab1.size() < batchszab;
                        ++ibab) {
                    const schedule_group &grp = addschab.get_node(ibab);
                    std::set<size_t> blks;
                    for(typename schedule_group::const_iterator i = grp.begin();
                            i != grp.end(); ++i) {
                        blks.insert(i->cia);
                    }
                    batchab1.insert(batchab1.end(), blks.begin(), blks.end());
                }
            } else {
                for(; ibab != addschab.end() && batchab1.size() < batchszab;
                        ++ibab) {
                    const schedule_group &grp = addschab.get_node(ibab);
                    std::set<size_t> blks;
                    for(typename schedule_group::const_iterator i = grp.begin();
                            i != grp.end(); ++i) {
                        index<NAB> iab;
                        abs_index<NAB>::get_index(i->cia, bidimsab, iab);
                        iab.permute(permab1);
                        short_orbit<NAB, element_type> oab(symab10, iab);
                        blks.insert(oab.get_acindex());
                    }
                    batchab1.insert(batchab1.end(), blks.begin(), blks.end());
                }
            }
            if(batchab1.size() == 0) continue;

            // Compute batch of AB
            gen_bto_aux_copy<NAB, Traits> ab1cout(symab1, btab1);
            tensor_transf<NAB, element_type> trab0;
            tensor_transf<NAB, element_type> trab1(sympermab,
                scalar_transf<element_type>(m_symmab ? 1.0 : -1.0));
            gen_bto_aux_symmetrize<NAB, Traits> ab1cout2(
                symab10, symab1, ab1cout);
            ab1cout2.add_transf(trab0);
            ab1cout2.add_transf(trab1);
            ab1cout.open();
            ab1cout2.open();
            compute_batch_ab(contr1,
                bidimsa, perma, symat, batchsza,
                bidimsb, permb, symbt, batchszb,
                bisab1, batchab1, ab1cout2);
            ab1cout2.close();
            ab1cout.close();

            {
                tensor_transf<NAB, element_type> trab(permab);
                gen_bto_aux_copy<NAB, Traits> cpabout(symab2, btab2);
                cpabout.open();
                gen_bto_copy_ab_type(btab1, trab).perform(cpabout);
                cpabout.close();
            }

            {
                gen_block_tensor_ctrl<NAB, bti_traits> cab2(btab2);
                cab2.req_nonzero_blocks(batchab2);
                cab2.req_symmetry().clear();
            }
            block_list<NAB> blab(bidimsab2, batchab2), blabx(bidimsab2);
            gen_bto_unfold_block_list<NAB, Traits>(symab2, blab).build(blabx);

            for(size_t ibc = 0; ibc < nblkc;) {

                batchc.clear();
                if(permc.is_identity()) {
                    for(; ibc < nblkc && batchc.size() < batchszc; ibc++) {
                        batchc.push_back(blstc[ibc]);
                    }
                } else {
                    for(; ibc < nblkc && batchc.size() < batchszc; ibc++) {
                        index<NC> ic;
                        abs_index<NC>::get_index(blstc[ibc], bidimsc, ic);
                        ic.permute(permc);
                        short_orbit<NC, element_type> oct(symct, ic);
                        batchc.push_back(oct.get_acindex());
                    }
                }

                if(batchc.size() == 0) continue;

                gen_bto_set_c_type(Traits::zero()).perform(btc2);
                {
                    tensor_transf<NC, element_type> trc(permc);
                    gen_bto_aux_copy<NC, Traits> cpcout(symc2, btc2);
                    cpcout.open();
                    gen_bto_copy_c_type(m_btc, trc).perform(batchc, cpcout);
                    cpcout.close();
                }
                {
                    gen_block_tensor_ctrl<NC, bti_traits> cc2(btc2);
                    cc2.req_nonzero_blocks(blstc2);
                    cc2.req_symmetry().clear();
                }
                block_list<NC> blc(bidimsc2, blstc2), blcx(bidimsc2);
                gen_bto_unfold_block_list<NC, Traits>(symc2, blc).build(blcx);

                typename assignment_schedule<ND, element_type>::iterator ibd =
                    m_schd.begin();
                while(ibd != m_schd.end()) {

                    batchd.clear();

                    for(; ibd != m_schd.end() && batchd.size() < batchszd;
                            ++ibd) {
                        index<ND> id;
                        abs_index<ND>::get_index(m_schd.get_abs_index(ibd),
                            bidimsd, id);
                        id.permute(permd);
                        short_orbit<ND, element_type> odt(symdt, id);
                        batchd.push_back(odt.get_acindex());
                    }
                    if(batchd.size() == 0) continue;

                    tensor_transf<ND, element_type> trd(permdinv);
                    gen_bto_aux_transform<ND, Traits> out2(trd,
                        m_symd.get_symmetry(), out);
                    out2.open();
                    gen_bto_contract2_batch<N1 + N2, N3, K2, Traits, Timed>(
                        contr2, btab1, btab2, permab, kab, blabx, batchab2,
                        m_btc, btc2, permc, m_kc, blcx, batchc,
                        symdt.get_bis(), m_kd).perform(batchd, out2);
                    out2.close();
                }
            }
        }

    } catch(...) {
        gen_bto_symcontract3::stop_timer();
        throw;
    }

    gen_bto_symcontract3::stop_timer();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_symcontract3<N1, N2, N3, K1, K2, Traits, Timed>::compute_batch_ab(
    const contraction2<N1, N2 + K2, K1> &contr,
    const dimensions<NA> &bidimsa,
    const permutation<NA> &perma,
    const symmetry<NA, element_type> &symat, size_t batchsza,
    const dimensions<NB> &bidimsb,
    const permutation<NB> &permb,
    const symmetry<NB, element_type> &symbt, size_t batchszb,
    const block_index_space<NAB> &bisab,
    const std::vector<size_t> &blst,
    gen_block_stream_i<NAB, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NA>::type
        temp_block_tensor_a_type;
    typedef typename Traits::template temp_block_tensor_type<NB>::type
        temp_block_tensor_b_type;
    typedef gen_bto_set<NA, Traits, Timed> gen_bto_set_a_type;
    typedef gen_bto_set<NB, Traits, Timed> gen_bto_set_b_type;
    typedef gen_bto_copy< NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, Traits, Timed> gen_bto_copy_b_type;

    gen_bto_symcontract3::start_timer("compute_batch_ab");

    try {

        //  Compute the number of non-zero blocks in A and B

        std::vector<size_t> blsta, blstb;

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            ca.req_nonzero_blocks(blsta);
            cb.req_nonzero_blocks(blstb);
        }

        size_t nblka = blsta.size(), nblkb = blstb.size();//, nblkab = 0;
        //nblkab = std::distance(m_schab.begin(), m_schab.end());

        scalar_transf<element_type> kab;

        block_index_space<NA> bisa2(m_bta.get_bis());
        bisa2.permute(perma);
        block_index_space<NB> bisb2(m_btb.get_bis());
        bisb2.permute(permb);
        dimensions<NA> bidimsa2 = bisa2.get_block_index_dims();
        dimensions<NB> bidimsb2 = bisb2.get_block_index_dims();

        temp_block_tensor_a_type bta2(bisa2);
        temp_block_tensor_b_type btb2(bisb2);

        symmetry<NA, element_type> syma2(bisa2);
        symmetry<NB, element_type> symb2(bisb2);

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            so_permute<NA, element_type>(ca.req_const_symmetry(), perma).
                perform(syma2);
        }
        {
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            so_permute<NB, element_type>(cb.req_const_symmetry(), permb).
                perform(symb2);
        }

        std::vector<size_t> blsta2, blstb2;

        //  Batching loops

        std::vector<size_t> batcha, batchb;
        batcha.reserve(batchsza);
        batchb.reserve(batchszb);

        for(size_t iba = 0; iba < nblka;) {

            batcha.clear();
            if(perma.is_identity()) {
                for(; iba < nblka && batcha.size() < batchsza; iba++) {
                    batcha.push_back(blsta[iba]);
                }
            } else {
                for(; iba < nblka && batcha.size() < batchsza; iba++) {
                    index<NA> ia;
                    abs_index<NA>::get_index(blsta[iba], bidimsa, ia);
                    ia.permute(perma);
                    short_orbit<NA, element_type> oat(symat, ia);
                    batcha.push_back(oat.get_acindex());
                }
            }

            if(batcha.size() == 0) continue;

            gen_bto_set_a_type(Traits::zero()).perform(bta2);
            {
                tensor_transf<NA, element_type> tra(perma);
                gen_bto_aux_copy<NA, Traits> cpaout(syma2, bta2);
                cpaout.open();
                gen_bto_copy_a_type(m_bta, tra).perform(batcha, cpaout);
                cpaout.close();
            }
            {
                gen_block_tensor_ctrl<NA, bti_traits> ca2(bta2);
                ca2.req_nonzero_blocks(blsta2);
                ca2.req_symmetry().clear();
            }
            block_list<NA> bla(bidimsa2, blsta2), blax(bidimsa2);
            gen_bto_unfold_block_list<NA, Traits>(syma2, bla).build(blax);
//            gen_bto_unfold_symmetry<NA, Traits>().perform(bta2);

            for(size_t ibb = 0; ibb < nblkb;) {

                batchb.clear();
                if(permb.is_identity()) {
                    for(; ibb < nblkb && batchb.size() < batchszb; ibb++) {
                        batchb.push_back(blstb[ibb]);
                    }
                } else {
                    for(; ibb < nblkb && batchb.size() < batchszb; ibb++) {
                        index<NB> ib;
                        abs_index<NB>::get_index(blstb[ibb], bidimsb, ib);
                        ib.permute(permb);
                        short_orbit<NB, element_type> obt(symbt, ib);
                        batchb.push_back(obt.get_acindex());
                    }
                }

                if(batchb.size() == 0) continue;

                gen_bto_set_b_type(Traits::zero()).perform(btb2);
                {
                    tensor_transf<NB, element_type> trb(permb);
                    gen_bto_aux_copy<NB, Traits> cpbout(symb2, btb2);
                    cpbout.open();
                    gen_bto_copy_b_type(m_btb, trb).perform(batchb, cpbout);
                    cpbout.close();
                }
                {
                    gen_block_tensor_ctrl<NB, bti_traits> cb2(btb2);
                    cb2.req_nonzero_blocks(blstb2);
                    cb2.req_symmetry().clear();
                }
                block_list<NB> blb(bidimsb2, blstb2), blbx(bidimsb2);
                gen_bto_unfold_block_list<NB, Traits>(symb2, blb).build(blbx);
//                gen_bto_unfold_symmetry<NB, Traits>().perform(btb2);

                gen_bto_contract2_batch<N1, N2 + K2, K1, Traits, Timed>(contr,
                    m_bta, bta2, perma, m_ka, blax, batcha,
                    m_btb, btb2, permb, m_kb, blbx, batchb, bisab, kab).
                    perform(blst, out);
            }
        }

    } catch(...) {
        gen_bto_symcontract3::stop_timer("compute_batch_ab");
        throw;
    }

    gen_bto_symcontract3::stop_timer("compute_batch_ab");
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_symcontract3<N1, N2, N3, K1, K2, Traits, Timed>::make_schedule() {

    gen_bto_symcontract3::start_timer("make_schedule");

    dimensions<NAB> bidimsab(m_symab.get_bis().get_block_index_dims());

    //  List of non-zero orbits before symmetrization of AB
    gen_bto_contract2_nzorb<N1, N2 + K2, K1, Traits> nzorb1(m_contr1,
        m_bta, m_btb, m_symab.get_symmetry0());
    nzorb1.build();

    //  List of non-zero orbits after symmetrization of AB
    assignment_schedule<NAB, element_type> schab_sym(bidimsab);

    //  Warning. This algorithm for forming schab_sym may have a bug in it
    const block_list<NAB> &blstab = nzorb1.get_blst();
    for(typename block_list<NAB>::iterator i = blstab.begin();
        i != blstab.end(); ++i) {

        m_schab.insert(blstab.get_abs_index(i));

        orbit<NAB, element_type> oab(m_symab.get_symmetry0(),
            blstab.get_abs_index(i));
        for(typename orbit<NAB, element_type>::iterator io = oab.begin();
            io != oab.end(); ++io) {

            size_t aiab1 = oab.get_abs_index(io);
            index<NAB> iab;
            abs_index<NAB>::get_index(aiab1, bidimsab, iab);
            short_orbit<NAB, element_type> oab1(m_symab.get_symmetry(), iab);
            iab.permute(m_sympermab);
            short_orbit<NAB, element_type> oab2(m_symab.get_symmetry(), iab);
            size_t aiab2 = abs_index<NAB>::get_abs_index(iab, bidimsab);

            if(oab1.get_acindex() == aiab1 && !schab_sym.contains(aiab1)) {
                schab_sym.insert(aiab1);
            }
            if(oab2.get_acindex() == aiab2 && !schab_sym.contains(aiab2)) {
                schab_sym.insert(aiab2);
            }
        }
    }

    //  List of nonzero orbits in the result of contraction of
    //  symmetrized AB with C
    gen_bto_contract2_nzorb<N1 + N2, N3, K2, Traits> nzorb2(m_contr2,
        m_symab.get_symmetry(), schab_sym, m_btc, m_symd.get_symmetry());
    nzorb2.build();

    const block_list<ND> &blstd = nzorb2.get_blst();
    for(typename block_list<ND>::iterator i = blstd.begin();
        i != blstd.end(); ++i) {
        m_schd.insert(blstd.get_abs_index(i));
    }

    gen_bto_symcontract3::stop_timer("make_schedule");
}



} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMCONTRACT3_IMPL_H

