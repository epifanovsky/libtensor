#ifndef LIBTENSOR_GEN_BTO_CONTRACT3_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT3_IMPL_H

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
#include "gen_bto_unfold_symmetry.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_copy.h"
#include "../gen_bto_aux_transform.h"
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

        size_t nblka = blsta.size(), nblkb = blstb.size(), nblkc = blstc.size(),
            nblkab = 0, nblkd = 0;
        nblkab = std::distance(m_schab.begin(), m_schab.end());
        nblkd = std::distance(m_schd.begin(), m_schd.end());

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0 || nblkc == 0 || nblkab == 0) {
            gen_bto_contract3::stop_timer();
            return;
        }

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
        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            so_permute<NA, element_type>(ca.req_const_symmetry(), perma).
                perform(symat);
            so_permute<NB, element_type>(cb.req_const_symmetry(), permb).
                perform(symbt);
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

        temp_block_tensor_ab_type btab1(bisab1);
        temp_block_tensor_d_type btdt(bisdt);

        gen_block_tensor_rd_ctrl<ND, bti_traits> cdt(btdt);

        //  Batching loops

        dimensions<NA> bidimsa(m_bta.get_bis().get_block_index_dims());
        dimensions<NB> bidimsb(m_btb.get_bis().get_block_index_dims());
        dimensions<NC> bidimsc(m_btc.get_bis().get_block_index_dims());
        dimensions<NAB> bidimsab(m_symab.get_bis().get_block_index_dims());
        dimensions<NAB> bidimsab1(symab1.get_bis().get_block_index_dims());
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

        typename assignment_schedule<NAB, element_type>::iterator ibab =
            m_schab.begin();
        while (ibab != m_schab.end()) {

            batchab1.clear();
            batchab2.clear();
            if (permab1.is_identity()) {
                for(; ibab != m_schab.end() &&
                        batchab1.size() < batchszab; ++ibab) {
                    batchab1.push_back(m_schab.get_abs_index(ibab));
                }
            }
            else {
                for(; ibab != m_schab.end() &&
                        batchab1.size() < batchszab; ++ibab) {

                    index<NAB> iab;
                    abs_index<NAB>::get_index(*ibab, bidimsab, iab);
                    iab.permute(permab1);
                    short_orbit<NAB, element_type> oab(symab1, iab);
                    batchab1.push_back(oab.get_acindex());
                }
            }
            if(batchab1.size() == 0) continue;

            // Compute batch of AB
            gen_bto_aux_copy<NAB, Traits> ab1cout(symab1, btab1);
            ab1cout.open();
            compute_batch_ab(contr1,
                    bidimsa, perma, symat, batchsza,
                    bidimsb, permb, symbt, batchszb,
                    m_symab.get_bis(), batchab1, ab1cout);
            ab1cout.close();

            if(!permab.is_identity()) {
                gen_block_tensor_rd_ctrl<NAB, bti_traits> cab1(btab1);
                cab1.req_nonzero_blocks(batchab1);
                for (size_t i = 0; i < batchab1.size(); i++) {
                    index<NAB> iab;
                    abs_index<NAB>::get_index(batchab1[i], bidimsab1, iab);
                    iab.permute(permab);
                    short_orbit<NAB, element_type> oab(symab2, iab);
                    batchab2.push_back(oab.get_acindex());
                }
            } else {
                gen_block_tensor_rd_ctrl<NAB, bti_traits> cab1(btab1);
                cab1.req_nonzero_blocks(batchab2);
            }

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
                        contr2, btab1, permab, kab, batchab2,
                        m_btc, permc, m_kc, batchc,
                        symdt.get_bis(), m_kd).perform(batchd, out2);
                    out2.close();
                }
            }
        }

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
    typedef gen_bto_copy< NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, Traits, Timed> gen_bto_copy_b_type;

    gen_bto_contract3::start_timer("compute_batch_ab");

    try {

        //  Compute the number of non-zero blocks in A and B

        std::vector<size_t> blsta, blstb;

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            ca.req_nonzero_blocks(blsta);
            cb.req_nonzero_blocks(blstb);
        }

        size_t nblka = blsta.size(), nblkb = blstb.size(), nblkab = 0;
        nblkab = std::distance(m_schab.begin(), m_schab.end());

        scalar_transf<element_type> kab;

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

                gen_bto_contract2_batch<N1, N2 + K2, K1, Traits, Timed>(contr,
                    m_bta, perma, m_ka, batcha,
                    m_btb, permb, m_kb, batchb, bisab, kab).
                    perform(blst, out);
            }
        }

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

    gen_bto_contract2_nzorb<N1, N2 + K2, K1, Traits> nzorb1(m_contr1,
        m_bta, m_btb, m_symab.get_symmetry());

    nzorb1.build();
    const block_list<NAB> &blstab = nzorb1.get_blst();
    for(typename block_list<NAB>::iterator i = blstab.begin();
            i != blstab.end(); ++i) {
        m_schab.insert(blstab.get_abs_index(i));
    }

    gen_bto_contract2_nzorb<N1 + N2, N3, K2, Traits> nzorb2(m_contr2,
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

