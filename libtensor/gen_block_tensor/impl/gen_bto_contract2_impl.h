#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H

#include <iterator>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_contract2_align.h"
#include "gen_bto_contract2_batch_impl.h"
#include "gen_bto_contract2_batching_policy.h"
#include "gen_bto_contract2_clst_builder.h"
#include "gen_bto_contract2_nzorb.h"
#include "gen_bto_contract2_sym_impl.h"
#include "gen_bto_prefetch.h"
#include "gen_bto_set_impl.h"
#include "gen_bto_unfold_block_list.h"
#include "gen_bto_unfold_symmetry.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_transform.h"
#include "../gen_bto_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2<N, M, K, Traits, Timed>::gen_bto_contract2(
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
void gen_bto_contract2<N, M, K, Traits, Timed>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NA>::type
        temp_block_tensor_a_type;
    typedef typename Traits::template temp_block_tensor_type<NB>::type
        temp_block_tensor_b_type;
    typedef gen_bto_set<NA, Traits, Timed> gen_bto_set_a_type;
    typedef gen_bto_set<NB, Traits, Timed> gen_bto_set_b_type;
    typedef gen_bto_copy<NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy<NB, Traits, Timed> gen_bto_copy_b_type;

    gen_bto_contract2::start_timer();

    try {

        //  Compute the number of non-zero blocks in A and B

        std::vector<size_t> blsta, blstb;

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            ca.req_nonzero_blocks(blsta);
            cb.req_nonzero_blocks(blstb);
        }

        size_t nblka = blsta.size(), nblkb = blstb.size(), nblkc = 0;
        nblkc = std::distance(m_sch.begin(), m_sch.end());

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0) {
            gen_bto_contract2::stop_timer();
            return;
        }

        //  Compute optimal permutations of A, B, and C

        gen_bto_contract2_align<N, M, K> align(m_contr);
        const permutation<NA> &perma = align.get_perma();
        const permutation<NB> &permb = align.get_permb();
        const permutation<NC> &permc = align.get_permc();
        permutation<NC> permcinv(permc, true);

        //  Prepare permuted arguments

        contraction2<N, M, K> contr(m_contr);
        contr.permute_a(perma);
        contr.permute_b(permb);
        contr.permute_c(permc);

        block_index_space<NA> bisat(m_bta.get_bis());
        bisat.permute(perma);
        block_index_space<NB> bisbt(m_btb.get_bis());
        bisbt.permute(permb);
        block_index_space<NC> bisct(m_symc.get_bis());
        bisct.permute(permc);

        symmetry<NA, element_type> symat(bisat);
        symmetry<NB, element_type> symbt(bisbt);
        symmetry<NC, element_type> symct(bisct);
        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            so_permute<NA, element_type>(ca.req_const_symmetry(), perma).
                perform(symat);
            so_permute<NB, element_type>(cb.req_const_symmetry(), permb).
                perform(symbt);
            so_permute<NC, element_type>(m_symc.get_symmetry(), permc).
                perform(symct);
        }

        //  Batching loops

        dimensions<NA> bidimsa(m_bta.get_bis().get_block_index_dims());
        dimensions<NB> bidimsb(m_btb.get_bis().get_block_index_dims());
        dimensions<NC> bidimsc(m_symc.get_bis().get_block_index_dims());
        dimensions<NC> bidimsct(bisct.get_block_index_dims());

        gen_bto_contract2_batching_policy<N, M, K> bp(m_contr,
            nblka, nblkb, nblkc);
        size_t batchsza = bp.get_bsz_a(), batchszb = bp.get_bsz_b(),
            batchszc = bp.get_bsz_c();

        std::list< std::vector<size_t> > batchesa, batchesb, batchesc,
            fbatchesa, fbatchesb;
        typedef typename std::list< std::vector<size_t> >::const_iterator
            batch_iterator;

        for(size_t iba = 0; iba < nblka;) {

            batchesa.push_back(std::vector<size_t>());
            fbatchesa.push_back(std::vector<size_t>());
            std::vector<size_t> &batcha = batchesa.back();
            std::vector<size_t> &fbatcha = fbatchesa.back();
            batcha.reserve(batchsza);
            fbatcha.reserve(batchsza);

            if(perma.is_identity()) {
                for(; iba < nblka && batcha.size() < batchsza; iba++) {
                    batcha.push_back(blsta[iba]);
                    fbatcha.push_back(blsta[iba]);
                }
            } else {
                for(; iba < nblka && batcha.size() < batchsza; iba++) {
                    index<NA> ia;
                    abs_index<NA>::get_index(blsta[iba], bidimsa, ia);
                    ia.permute(perma);
                    short_orbit<NA, element_type> oat(symat, ia);
                    batcha.push_back(oat.get_acindex());
                    fbatcha.push_back(blsta[iba]);
                }
            }
        }

        for(size_t ibb = 0; ibb < nblkb;) {

            batchesb.push_back(std::vector<size_t>());
            fbatchesb.push_back(std::vector<size_t>());
            std::vector<size_t> &batchb = batchesb.back();
            std::vector<size_t> &fbatchb = fbatchesb.back();
            batchb.reserve(batchszb);
            fbatchb.reserve(batchszb);

            if(permb.is_identity()) {
                for(; ibb < nblkb && batchb.size() < batchszb; ibb++) {
                    batchb.push_back(blstb[ibb]);
                    fbatchb.push_back(blstb[ibb]);
                }
            } else {
                for(; ibb < nblkb && batchb.size() < batchszb; ibb++) {
                    index<NB> ib;
                    abs_index<NB>::get_index(blstb[ibb], bidimsb, ib);
                    ib.permute(permb);
                    short_orbit<NB, element_type> obt(symbt, ib);
                    batchb.push_back(obt.get_acindex());
                    fbatchb.push_back(blstb[ibb]);
                }
            }
        }

        typename assignment_schedule<NC, element_type>::iterator ibc =
            m_sch.begin();
        while(ibc != m_sch.end()) {

            batchesc.push_back(std::vector<size_t>());
            std::vector<size_t> &batchc = batchesc.back();
            batchc.reserve(batchszc);

            for(; ibc != m_sch.end() && batchc.size() < batchszc; ++ibc) {
                index<NC> ic;
                abs_index<NC>::get_index(m_sch.get_abs_index(ibc), bidimsc, ic);
                ic.permute(permc);
                short_orbit<NC, element_type> oct(symct, ic);
                batchc.push_back(oct.get_acindex());
            }
        }

        gen_bto_prefetch<NA, Traits> prefetch_a(m_bta);
        gen_bto_prefetch<NB, Traits> prefetch_b(m_btb);

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

        for(batch_iterator iba1 = batchesa.begin(), iba2 = fbatchesa.begin();
            iba1 != batchesa.end(); ++iba1, ++iba2) {

            const std::vector<size_t> &batcha = *iba1;

            gen_bto_set_a_type(Traits::zero()).perform(bta2);
            {
                tensor_transf<NA, element_type> tra(perma);
                gen_bto_aux_copy<NA, Traits> cpaout(syma2, bta2, false);
                cpaout.open();
                gen_bto_copy_a_type(m_bta, tra).perform(batcha, cpaout);
                cpaout.close();
            }
            {
                gen_block_tensor_rd_ctrl<NA, bti_traits> ca2(bta2);
                ca2.req_nonzero_blocks(blsta2);
            }
            block_list<NA> bla(bidimsa2, blsta2), blax(bidimsa2);
            gen_bto_unfold_block_list<NA, Traits>(syma2, bla).build(blax);
//            gen_bto_unfold_symmetry<NA, Traits>().perform(bta2);

            for(batch_iterator ibb1 = batchesb.begin(),
                ibb2 = fbatchesb.begin(); ibb1 != batchesb.end();
                ++ibb1, ++ibb2) {

                const std::vector<size_t> &batchb = *ibb1;

                gen_bto_set_b_type(Traits::zero()).perform(btb2);
                {
                    tensor_transf<NB, element_type> trb(permb);
                    gen_bto_aux_copy<NB, Traits> cpbout(symb2, btb2, false);
                    cpbout.open();
                    gen_bto_copy_b_type(m_btb, trb).perform(batchb, cpbout);
                    cpbout.close();
                }
                {
                    gen_block_tensor_rd_ctrl<NB, bti_traits> cb2(btb2);
                    cb2.req_nonzero_blocks(blstb2);
                }
                block_list<NB> blb(bidimsb2, blstb2), blbx(bidimsb2);
                gen_bto_unfold_block_list<NB, Traits>(symb2, blb).build(blbx);
//                gen_bto_unfold_symmetry<NB, Traits>().perform(btb2);

                batch_iterator iba3 = iba2, ibb3 = ibb2;
                ++iba3; ++ibb3;
                if(ibb3 != fbatchesb.end()) {
                    prefetch_b.perform(*ibb3);
                } else {
                    if(iba3 != fbatchesa.end()) prefetch_a.perform(*iba3);
                    prefetch_b.perform(fbatchesb.front());
                }

                for(batch_iterator ibc = batchesc.begin();
                    ibc != batchesc.end(); ++ibc) {

                    const std::vector<size_t> &batchc = *ibc;

                    tensor_transf<NC, element_type> trc(permcinv);
                    gen_bto_aux_transform<NC, Traits> out2(trc,
                        m_symc.get_symmetry(), out);
                    out2.open();
                    gen_bto_contract2_batch<N, M, K, Traits, Timed>(contr,
                        m_bta, bta2, perma, m_ka, blax, batcha,
                        m_btb, btb2, permb, m_kb, blbx, batchb,
                        symct.get_bis(), m_kc).perform(batchc, out2);
                    out2.close();
                }
            }
        }

    } catch(...) {
        gen_bto_contract2::stop_timer();
        throw;
    }

    gen_bto_contract2::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::compute_block(
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
void gen_bto_contract2<N, M, K, Traits, Timed>::make_schedule() {

    gen_bto_contract2::start_timer("make_schedule");

    gen_bto_contract2_nzorb<N, M, K, Traits> nzorb(m_contr, m_bta, m_btb,
        m_symc.get_symmetry());

    nzorb.build();
    const block_list<NC> &blstc = nzorb.get_blst();
    for(typename block_list<NC>::iterator i = blstc.begin();
            i != blstc.end(); ++i) {
        m_sch.insert(blstc.get_abs_index(i));
    }

    gen_bto_contract2::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
