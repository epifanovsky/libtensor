#ifndef LIBTENSOR_GEN_BTO_CONTRACT3_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT3_IMPL_H

#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_copy_impl.h"
#include "gen_bto_contract2_align.h"
#include "gen_bto_contract2_batch_impl.h"
#include "gen_bto_contract2_clst_builder_impl.h"
#include "gen_bto_contract2_nzorb_impl.h"
#include "gen_bto_contract2_sym_impl.h"
#include "gen_bto_contract3_batching_policy.h"
#include "gen_bto_unfold_symmetry.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_add.h"
#include "../gen_bto_aux_copy.h"
#include "../gen_bto_contract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
gen_bto_contract3<N1, N2, N3, K1, K2, Traits, Timed>::gen_bto_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const scalar_transf<element_type> &ka,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const scalar_transf<element_type> &kb,
    gen_block_tensor_rd_i<NC, bti_traits> &btc,
    const scalar_transf<element_type> &kc,
    const scalar_transf<element_type> &kd) :

    m_contr1(contr1), m_contr2(contr2), m_bta(bta), m_ka(ka),
    m_btb(btb), m_kb(kb), m_btc(btc), m_kc(kc), m_kd(kd),
    m_symab(contr1, bta, btb),
    m_symd(contr2, m_symab.get_symmetry(), retrieve_symmetry(btc)),
    m_schab(m_symab.get_bis().get_block_index_dims()),
    m_schd(m_symd.get_bis().get_block_index_dims()) {

    make_schedule();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_contract3<N1, N2, N3, K1, K2, Traits, Timed>::perform(
    gen_block_stream_i<ND, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
            temp_block_tensor_c_type;
    typedef typename Traits::template temp_block_tensor_type<ND>::type
            temp_block_tensor_d_type;
    typedef typename Traits::template temp_block_tensor_type<NAB>::type
            temp_block_tensor_ab_type;

    typedef gen_bto_copy<NAB, Traits, Timed> gen_bto_copy_ab_type;
    typedef gen_bto_copy<NC, Traits, Timed> gen_bto_copy_c_type;
    typedef gen_bto_copy<ND, Traits, Timed> gen_bto_copy_d_type;

    gen_bto_contract3::start_timer();

    try {

        out.open();

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
        gen_block_tensor_rd_ctrl<NC, bti_traits> cc(m_btc);

        //  Compute the number of blocks in A, B, C, and D

        size_t nblka = 0, nblkb = 0, nblkc = 0, nblkab = 0, nblkd = 0;
        orbit_list<NA, element_type> ola(ca.req_const_symmetry());
        for(typename orbit_list<NA, element_type>::iterator ioa = ola.begin();
                ioa != ola.end(); ++ioa) {
            if(!ca.req_is_zero_block(ola.get_index(ioa))) nblka++;
        }
        orbit_list<NB, element_type> olb(cb.req_const_symmetry());
        for(typename orbit_list<NB, element_type>::iterator iob = olb.begin();
                iob != olb.end(); ++iob) {
            if(!cb.req_is_zero_block(olb.get_index(iob))) nblkb++;
        }
        orbit_list<NC, element_type> olc(cc.req_const_symmetry());
        for(typename orbit_list<NC, element_type>::iterator ioc = olc.begin();
                ioc != olc.end(); ++ioc) {
            if(!cc.req_is_zero_block(olc.get_index(ioc))) nblkc++;
        }
        for(typename assignment_schedule<NAB, element_type>::iterator isch =
                m_schab.begin(); isch != m_schab.end(); ++isch) {
            nblkab++;
        }
        for(typename assignment_schedule<ND, element_type>::iterator isch =
                m_schd.begin(); isch != m_schd.end(); ++isch) {
            nblkd++;
        }

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0 || nblkc == 0 || nblkab == 0) {
            out.close();
            gen_bto_contract3::stop_timer();
            return;
        }

        gen_bto_contract3_batching_policy<N1, N2, N3, K1, K2> bp(
                m_contr1, m_contr2, nblka, nblkb, nblkc, nblkab, nblkd);

        //  Compute optimal permutations of A and B to perform 1st contraction

        gen_bto_contract2_align<N1, N2 + K2, K1> align1(m_contr1);

        const permutation<NA> &perma = align1.get_perma();
        const permutation<NB> &permb = align1.get_permb();
        const permutation<NAB> &permab1 = align1.get_permc();

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
        symmetry<NAB, element_type> symab1(bisab1);
        so_permute<NA, element_type>(
                ca.req_const_symmetry(), perma).perform(symat);
        so_permute<NB, element_type>(
                cb.req_const_symmetry(), permb).perform(symbt);
        so_permute<NAB, element_type>(
                m_symab.get_symmetry(), permab1).perform(symab1);

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
        so_permute<NAB, element_type>(m_symab.get_symmetry(), permab2).
            perform(symab2);
        so_permute<NC, element_type>(cc.req_const_symmetry(), permc).
            perform(symct);
        so_permute<ND, element_type>(m_symd.get_symmetry(), permd).
            perform(symdt);

        //  Temporary partial AB1, AB2, C, and D

        temp_block_tensor_ab_type btab1(bisab1), btab2(bisab2), btab3(bisab2);
        temp_block_tensor_c_type btct(bisct), btct3(bisct);
        temp_block_tensor_d_type btdt(bisdt);

        gen_block_tensor_rd_ctrl<ND, bti_traits> cdt(btdt);
        gen_block_tensor_ctrl<NAB, bti_traits> cab(btab1);

        dimensions<NAB> bidimsab(m_symab.get_bis().get_block_index_dims());
        dimensions<NAB> bidimsab1(symab1.get_bis().get_block_index_dims());
        dimensions<ND> bidimsd(m_symd.get_bis().get_block_index_dims());
        dimensions<ND> bidimsdt(bisdt.get_block_index_dims());

        scalar_transf<element_type> kab;

        std::vector<size_t> batchab1, batchab2, batchc, batchd1, batchd2;
        batchab1.reserve(bp.get_bsz_ab());
        batchab2.reserve(bp.get_bsz_ab());
        batchc.reserve(bp.get_bsz_c());
        batchd1.reserve(bp.get_bsz_d());
        batchd2.reserve(bp.get_bsz_d());

        typename assignment_schedule<NAB, element_type>::iterator ibab =
            m_schab.begin();
        while (ibab != m_schab.end()) {

            batchab1.clear();
            batchab2.clear();
            if (permab1.is_identity()) {
                for(; ibab != m_schab.end() &&
                        batchab1.size() < bp.get_bsz_ab(); ++ibab) {
                    batchab1.push_back(m_schab.get_abs_index(ibab));
                }
            }
            else {
                for(; ibab != m_schab.end() &&
                        batchab1.size() < bp.get_bsz_ab(); ++ibab) {

                    index<NAB> iab;
                    abs_index<NAB>::get_index(*ibab, bidimsab, iab);
                    iab.permute(permab1);
                    orbit<NAB, element_type> oab(symab1, iab, false);
                    batchab1.push_back(oab.get_acindex());
                }
            }
            if(batchab1.size() == 0) continue;

            // Compute batch of AB
            gen_bto_aux_copy<NAB, Traits> ab1cout(symab1, btab1);
            compute_batch_ab(contr1,
                    ola, perma, symat, bp.get_bsz_a(),
                    olb, permb, symbt, bp.get_bsz_b(),
                    m_symab.get_bis(), batchab1, ab1cout);

            bool use_orig_ab = permab.is_identity();
            if (! use_orig_ab) {
                gen_bto_contract3::start_timer("copy_ab");
                for (size_t i = 0; i < batchab1.size(); i++) {
                    index<NAB> iab;
                    abs_index<NAB>::get_index(batchab1[i], bidimsab1, iab);
                    if(! cab.req_is_zero_block(iab)) {
                        iab.permute(permab);
                        orbit<NAB, element_type> oab(symab2, iab, false);
                        batchab2.push_back(oab.get_acindex());
                    }
                }
                tensor_transf<NAB, element_type> trab(permab);
                gen_bto_aux_copy<NAB, Traits> ab2cout(symab2, btab2);
                gen_bto_copy_ab_type(btab1, trab).perform(batchab2, ab2cout);
                cab.req_zero_all_blocks();
                gen_bto_contract3::stop_timer("copy_ab");
            }

            gen_block_tensor_rd_i<NAB, bti_traits> &btabx =
                (use_orig_ab ? btab1 : btab2);

            {
                gen_bto_contract3::start_timer("copy_ab_2");
                tensor_transf<NAB, element_type> trab0;
                gen_bto_aux_copy<NAB, Traits> ab3cout(symab2, btab3);
                gen_bto_copy_ab_type(btabx, trab0).perform(ab3cout);
                gen_bto_unfold_symmetry<NAB, Traits>().perform(btab3);
                gen_bto_contract3::stop_timer("copy_ab_2");
            }

            typename orbit_list<NC, element_type>::iterator ioc = olc.begin();
            bool first_batch_c = true;
            while (ioc != olc.end()) {

                batchc.clear();
                if (permc.is_identity()) {
                    for (; ioc != olc.end() && batchc.size() < bp.get_bsz_c();
                            ++ioc) {
                        const index<NC> &ic = olc.get_index(ioc);
                        if(cc.req_is_zero_block(ic)) continue;
                        batchc.push_back(olc.get_abs_index(ioc));
                    }
                } else {
                    for(; ioc != olc.end() && batchc.size() < bp.get_bsz_c();
                            ++ioc) {
                        index<NC> ic = olc.get_index(ioc);
                        if(cc.req_is_zero_block(ic)) continue;
                        ic.permute(permc);
                        orbit<NC, element_type> oct(symct, ic, false);
                        batchc.push_back(oct.get_acindex());
                    }
                }

                bool use_orig_c = (first_batch_c &&
                        ioc == olc.end() && permc.is_identity());
                first_batch_c = false;

                if (! use_orig_c) {
                    gen_bto_contract3::start_timer("copy_c");
                    tensor_transf<NC, element_type> trc(permc);
                    gen_bto_aux_copy<NC, Traits> cpcout(symct, btct);
                    gen_bto_copy_c_type(m_btc, trc).perform(batchc, cpcout);
                    gen_bto_contract3::stop_timer("copy_c");
                }

                gen_block_tensor_rd_i<NC, bti_traits> &btc =
                        (use_orig_c ? m_btc : btct);

                {
                    gen_bto_contract3::start_timer("copy_c_2");
                    tensor_transf<NC, element_type> trc0;
                    gen_bto_aux_copy<NC, Traits> c3cout(symct, btct3);
                    gen_bto_copy_c_type(btc, trc0).perform(c3cout);
                    gen_bto_unfold_symmetry<NC, Traits>().perform(btct3);
                    gen_bto_contract3::stop_timer("copy_c_2");
                }

                if(batchc.size() == 0) continue;

                typename assignment_schedule<ND, element_type>::iterator ibd =
                    m_schd.begin();
                while(ibd != m_schd.end()) {

                    batchd1.clear();
                    batchd2.clear();

                    for(; ibd != m_schd.end() && batchd1.size() < bp.get_bsz_d();
                            ++ibd) {
                        index<ND> id;
                        abs_index<ND>::get_index(m_schd.get_abs_index(ibd),
                            bidimsd, id);
                        id.permute(permd);
                        orbit<ND, element_type> odt(symdt, id, false);
                        batchd1.push_back(odt.get_acindex());
                    }
                    if(batchd1.size() == 0) continue;

                    //  Calling this may break the symmetry of final result
                    //  in some cases, e.g. self-contraction
                    gen_bto_aux_copy<ND, Traits> dtcout(symdt, btdt);
                    gen_bto_contract2_batch<N1 + N2, N3, K2, Traits, Timed>(
                        contr2, btabx, btab3, kab, btc, btct3, m_kc,
                        symdt.get_bis(), m_kd).perform(batchd1, dtcout);

                    gen_bto_contract3::start_timer("copy_d");
                    for(size_t i = 0; i < batchd1.size(); i++) {
                        index<ND> id;
                        abs_index<ND>::get_index(batchd1[i], bidimsdt, id);
                        if(!cdt.req_is_zero_block(id)) {
                            id.permute(permdinv);
                            orbit<ND, element_type> od(
                                    m_symd.get_symmetry(), id, false);
                            batchd2.push_back(od.get_acindex());
                        }
                    }
                    tensor_transf<ND, element_type> trd(permdinv);
                    gen_bto_copy_d_type(btdt, trd).perform(batchd2, out);
                    gen_bto_contract3::stop_timer("copy_d");
                }
            }
        }

        out.close();

    } catch(...) {
        gen_bto_contract3::stop_timer();
        throw;
    }

    gen_bto_contract3::stop_timer();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_contract3<N1, N2, N3, K1, K2, Traits, Timed>::compute_batch_ab(
        const contraction2<N1, N2 + K2, K1> &contr,
        const orbit_list<NA, element_type> &ola, const permutation<NA> &perma,
        const symmetry<NA, element_type> &syma, size_t batchsza,
        const orbit_list<NB, element_type> &olb, const permutation<NB> &permb,
        const symmetry<NB, element_type> &symb, size_t batchszb,
        const block_index_space<NAB> &bisab,
        const std::vector<size_t> &blst,
        gen_block_stream_i<NAB, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NA>::type
            temp_block_tensor_a_type;
    typedef typename Traits::template temp_block_tensor_type<NB>::type
            temp_block_tensor_b_type;
    typedef gen_bto_copy< NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, Traits, Timed> gen_bto_copy_b_type;

    gen_bto_contract3::start_timer("compute_batch_ab");

    try {

        out.open();

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        temp_block_tensor_a_type btat(syma.get_bis()), btat2(syma.get_bis());
        temp_block_tensor_b_type btbt(symb.get_bis()), btbt2(symb.get_bis());

        scalar_transf<element_type> kab;

        //  Batching loops

        std::vector<size_t> batcha, batchb;
        batcha.reserve(batchsza);
        batchb.reserve(batchszb);

        typename orbit_list<NA, element_type>::iterator ioa = ola.begin();
        bool first_batch_a = true;
        while (ioa != ola.end()) {

            batcha.clear();
            if (perma.is_identity()) {
                for(; ioa != ola.end() && batcha.size() < batchsza; ++ioa) {
                    const index<NA> &ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    batcha.push_back(ola.get_abs_index(ioa));
                }
            } else {
                for (; ioa != ola.end() && batcha.size() < batchsza; ++ioa) {
                    index<NA> ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    ia.permute(perma);
                    orbit<NA, element_type> oat(syma, ia, false);
                    batcha.push_back(oat.get_acindex());
                }
            }

            //  If A need not be permuted and fits in one batch entirely,
            //  do not make a copy; use the original tensor
            bool use_orig_a = (first_batch_a && ioa == ola.end() &&
                perma.is_identity());
            first_batch_a = false;

            if (!use_orig_a) {
                gen_bto_contract3::start_timer("copy_a");
                tensor_transf<NA, element_type> tra(perma);
                gen_bto_aux_copy<NA, Traits> cpaout(syma, btat);
                gen_bto_copy_a_type(m_bta, tra).perform(batcha, cpaout);
                gen_bto_contract3::stop_timer("copy_a");
            }

            gen_block_tensor_rd_i<NA, bti_traits> &bta =
                    (use_orig_a ? m_bta : btat);

            {
                gen_bto_contract3::start_timer("copy_a_2");
                tensor_transf<NA, element_type> tra0;
                gen_bto_aux_copy<NA, Traits> cpa2out(syma, btat2);
                gen_bto_copy_a_type(bta, tra0).perform(cpa2out);
                gen_bto_unfold_symmetry<NA, Traits>().perform(btat2);
                gen_bto_contract3::stop_timer("copy_a_2");
            }

            if (batcha.size() == 0) continue;

            typename orbit_list<NB, element_type>::iterator iob = olb.begin();
            bool first_batch_b = true;
            while (iob != olb.end()) {

                batchb.clear();
                if (permb.is_identity()) {
                    for (; iob != olb.end() && batchb.size() < batchszb;
                            ++iob) {
                        const index<NB> &ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        batchb.push_back(olb.get_abs_index(iob));
                    }
                } else {
                    for(; iob != olb.end() && batchb.size() < batchszb;
                            ++iob) {
                        index<NB> ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        ib.permute(permb);
                        orbit<NB, element_type> obt(symb, ib, false);
                        batchb.push_back(obt.get_acindex());
                    }
                }

                bool use_orig_b = (first_batch_b &&
                        iob == olb.end() && permb.is_identity());
                first_batch_b = false;

                if(!use_orig_b) {
                    gen_bto_contract3::start_timer("copy_b");
                    tensor_transf<NB, element_type> trb(permb);
                    gen_bto_aux_copy<NB, Traits> cpbout(symb, btbt);
                    gen_bto_copy_b_type(m_btb, trb).perform(batchb, cpbout);
                    gen_bto_contract3::stop_timer("copy_b");
                }

                gen_block_tensor_rd_i<NB, bti_traits> &btb =
                        (use_orig_b ? m_btb : btbt);

                {
                    gen_bto_contract3::start_timer("copy_b_2");
                    tensor_transf<NB, element_type> trb0;
                    gen_bto_aux_copy<NB, Traits> cpb2out(symb, btbt2);
                    gen_bto_copy_b_type(btb, trb0).perform(cpb2out);
                    gen_bto_unfold_symmetry<NB, Traits>().perform(btbt2);
                    gen_bto_contract3::stop_timer("copy_b_2");
                }

                if(batchb.size() == 0) continue;

                //  Calling this may break the symmetry of final result
                //  in some cases, e.g. self-contraction
                gen_bto_contract2_batch<N1, N2 + K2, K1, Traits, Timed>(contr,
                    bta, btat2, m_ka, btb, btbt2, m_kb, bisab, kab).
                    perform(blst, out);

            }
        }

        out.close();

    } catch(...) {
        gen_bto_contract3::stop_timer("compute_batch_ab");
        throw;
    }

    gen_bto_contract3::stop_timer("compute_batch_ab");
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
void gen_bto_contract3<N1, N2, N3, K1, K2, Traits, Timed>::make_schedule() {

    gen_bto_contract3::start_timer("make_schedule");

    gen_bto_contract2_nzorb<N1, N2 + K2, K1, Traits, Timed> nzorb1(m_contr1,
        m_bta, m_btb, m_symab.get_symmetry());

    nzorb1.build();
    const block_list<NAB> &blstab = nzorb1.get_blst();
    for(typename block_list<NAB>::iterator i = blstab.begin();
            i != blstab.end(); ++i) {
        m_schab.insert(blstab.get_abs_index(i));
    }

    gen_bto_contract2_nzorb<N1 + N2, N3, K2, Traits, Timed> nzorb2(m_contr2,
            m_symab.get_symmetry(), m_schab, m_btc, m_symd.get_symmetry());

    nzorb2.build();
    const block_list<ND> &blstd = nzorb2.get_blst();
    for (typename block_list<ND>::iterator i = blstd.begin();
            i != blstd.end(); ++i) {
        m_schd.insert(blstd.get_abs_index(i));
    }

    gen_bto_contract3::stop_timer("make_schedule");
}



} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT3_IMPL_H

