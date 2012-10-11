#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H

#include <map>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_copy_impl.h"
#include "gen_bto_contract2_clst_impl.h"
#include "gen_bto_contract2_nzorb_impl.h"
#include "gen_bto_contract2_sym_impl.h"
#include "gen_bto_contract2_batch_impl.h"
#include "gen_bto_contract2_batching_policy.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_add.h"
#include "../gen_bto_aux_copy.h"
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
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bisc().get_block_index_dims()),
    m_sch(m_bidimsc) {

    make_schedule();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::perform(
        gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NA>::type
        temp_block_tensor_a_type;
    typedef typename Traits::template temp_block_tensor_type<NB>::type
        temp_block_tensor_b_type;
    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_c_type;

    typedef gen_bto_copy< NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, Traits, Timed> gen_bto_copy_b_type;
    typedef gen_bto_copy< NC, Traits, Timed> gen_bto_copy_c_type;

    gen_bto_contract2::start_timer();

    try {

        out.open();

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        //  Compute the number of blocks in A and B

        size_t nblka = 0, nblkb = 0, nblkc = 0;
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
        for(typename assignment_schedule<NC, element_type>::iterator i =
                m_sch.begin(); i != m_sch.end(); ++i) {
            nblkc++;
        }

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0) {
            out.close();
            gen_bto_contract2::stop_timer();
            return;
        }

        gen_bto_contract2_batching_policy<N, M, K, Traits> bp(m_contr,
                nblka, nblkb, nblkc);

        //  Compute optimal permutations of A, B, and C

        permutation<NA> perma;
        permutation<NB> permb;
        permutation<NC> permc;
        align(m_contr.get_conn(), perma, permb, permc);
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
        block_index_space<NC> bisct(m_symc.get_bisc());
        bisct.permute(permc);

        symmetry<NA, element_type> symat(bisat);
        symmetry<NB, element_type> symbt(bisbt);
        symmetry<NC, element_type> symct(bisct);
        so_permute<NA, element_type>(
                ca.req_const_symmetry(), perma).perform(symat);
        so_permute<NB, element_type>(
                cb.req_const_symmetry(), permb).perform(symbt);
        so_permute<NC, element_type>(
                m_symc.get_symc(), permc).perform(symct);

        //  Temporary partial A, B, and C

        temp_block_tensor_a_type btat(bisat);
        temp_block_tensor_b_type btbt(bisbt);
        temp_block_tensor_c_type btct(bisct);

        gen_block_tensor_rd_ctrl<NC, bti_traits> cct(btct);

        //  Batching loops

        dimensions<NC> bidimsc(m_symc.get_bisc().get_block_index_dims());
        dimensions<NC> bidimsct(bisct.get_block_index_dims());

        std::vector<size_t> batcha, batchb, batchc1, batchc2;
        batcha.reserve(bp.get_bsz_a());
        batchb.reserve(bp.get_bsz_b());
        batchc1.reserve(bp.get_bsz_c());
        batchc2.reserve(bp.get_bsz_c());

        typename orbit_list<NA, element_type>::iterator ioa = ola.begin();
        bool first_batch_a = true;
        while(ioa != ola.end()) {

            batcha.clear();
            if(perma.is_identity()) {
                for(; ioa != ola.end() && batcha.size() < bp.get_bsz_a();
                        ++ioa) {
                    const index<NA> &ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    batcha.push_back(ola.get_abs_index(ioa));
                }
            } else {
                for(; ioa != ola.end() && batcha.size() < bp.get_bsz_a();
                        ++ioa) {
                    index<NA> ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    ia.permute(perma);
                    orbit<NA, element_type> oat(symat, ia, false);
                    batcha.push_back(oat.get_acindex());
                }
            }

            //  If A need not be permuted and fits in one batch entirely,
            //  do not make a copy; use the original tensor
            bool use_orig_a = (first_batch_a && ioa == ola.end() &&
                perma.is_identity());
            first_batch_a = false;

            if(!use_orig_a) {
                gen_bto_contract2::start_timer("copy_a");
                tensor_transf<NA, element_type> tra(perma);
                gen_bto_aux_copy<NA, Traits> cpaout(symat, btat);
                gen_bto_copy_a_type(m_bta, tra).perform(batcha, cpaout);
                gen_bto_contract2::stop_timer("copy_a");
            }

            gen_block_tensor_rd_i<NA, bti_traits> &bta =
                    (use_orig_a ? m_bta : btat);

            if(batcha.size() == 0) continue;

            typename orbit_list<NB, element_type>::iterator iob = olb.begin();
            bool first_batch_b = true;
            while(iob != olb.end()) {

                batchb.clear();
                if(permb.is_identity()) {
                    for(; iob != olb.end() && batchb.size() < bp.get_bsz_b();
                            ++iob) {
                        const index<NB> &ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        batchb.push_back(olb.get_abs_index(iob));
                    }
                } else {
                    for(; iob != olb.end() && batchb.size() < bp.get_bsz_b();
                            ++iob) {
                        index<NB> ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        ib.permute(permb);
                        orbit<NB, element_type> obt(symbt, ib, false);
                        batchb.push_back(obt.get_acindex());
                    }
                }

                bool use_orig_b = (first_batch_b &&
                        iob == olb.end() && permb.is_identity());
                first_batch_b = false;

                if(!use_orig_b) {
                    gen_bto_contract2::start_timer("copy_b");
                    tensor_transf<NB, element_type> trb(permb);
                    gen_bto_aux_copy<NB, Traits> cpbout(symbt, btbt);
                    gen_bto_copy_b_type(m_btb, trb).perform(batchb, cpbout);
                    gen_bto_contract2::stop_timer("copy_b");
                }

                gen_block_tensor_rd_i<NB, bti_traits> &btb =
                        (use_orig_b ? m_btb : btbt);

                if(batchb.size() == 0) continue;

                typename assignment_schedule<NC, element_type>::iterator ibc =
                    m_sch.begin();
                while(ibc != m_sch.end()) {

                    batchc1.clear();
                    batchc2.clear();

                    for(; ibc != m_sch.end() && batchc1.size() < bp.get_bsz_c();
                            ++ibc) {
                        index<NC> ic;
                        abs_index<NC>::get_index(m_sch.get_abs_index(ibc),
                            bidimsc, ic);
                        ic.permute(permc);
                        orbit<NC, element_type> oct(symct, ic, false);
                        batchc1.push_back(oct.get_acindex());
                    }
                    if(batchc1.size() == 0) continue;

                    //  Calling this may break the symmetry of final result
                    //  in some cases, e.g. self-contraction
                    gen_bto_aux_copy<NC, Traits> ctcout(symct, btct);
                    gen_bto_contract2_batch<N, M, K, Traits, Timed> bto(contr,
                            bta, m_ka, btb, m_kb, symct.get_bis(), m_kc);
                    bto.perform(batchc1, ctcout);

                    gen_bto_contract2::start_timer("copy_c");
                    for(size_t i = 0; i < batchc1.size(); i++) {
                        index<NC> ic;
                        abs_index<NC>::get_index(batchc1[i], bidimsct, ic);
                        if(!cct.req_is_zero_block(ic)) {
                            ic.permute(permcinv);
                            orbit<NC, element_type> oc(
                                    m_symc.get_symc(), ic, false);
                            batchc2.push_back(oc.get_acindex());
                        }
                    }
                    tensor_transf<NC, element_type> trc(permcinv);
                    gen_bto_copy_c_type(btct, trc).perform(batchc2, out);
                    gen_bto_contract2::stop_timer("copy_c");
                }
            }
        }

        out.close();

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

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

    const symmetry<NA, element_type> &syma = ca.req_const_symmetry();
    const symmetry<NB, element_type> &symb = cb.req_const_symmetry();

    gen_bto_contract2_block<N, M, K, Traits, Timed> bto(m_contr, m_bta,
        syma, m_ka, m_btb, symb, m_kb, m_symc.get_bisc(), m_kc);

    bto.compute_block(zero, idxc, trc, blkc);

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::make_schedule() {

    gen_bto_contract2::start_timer("make_schedule");

    gen_bto_contract2_nzorb<N, M, K, Traits, Timed> nzorb(m_contr,
        m_bta, m_btb, m_symc.get_symc());

    nzorb.build();
    for(typename std::vector<size_t>::const_iterator i =
        nzorb.get_blst().begin(); i != nzorb.get_blst().end(); ++i) {
        m_sch.insert(*i);
    }

    gen_bto_contract2::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::align(
    const sequence<2 * (N + M + K), size_t> &conn,
    permutation<NA> &perma, permutation<NB> &permb,
    permutation<NC> &permc) {

    //  This algorithm reorders indexes in A, B, C so that the whole contraction
    //  can be done in a single matrix multiplication.
    //  Returned permutations perma, permb, permc need to be applied to
    //  the indexes of A, B, and C to get the matricized form.

    //  Numbering scheme:
    //  0     .. N - 1         -- outer indexes from A
    //  N     .. N + M - 1     -- outer indexes from B
    //  N + M .. N + M + K - 1 -- inner indexes

    size_t ioa = 0, iob = N, ii = NC;

    sequence<NA, size_t> idxa1(0), idxa2(0);
    sequence<NB, size_t> idxb1(0), idxb2(0);
    sequence<NC, size_t> idxc1(0), idxc2(0);

    //  Build initial index ordering

    for(size_t i = 0; i < NC; i++) {
        size_t j = conn[i] - NC;
        if(j < NA) {
            idxc1[i] = ioa;
            idxa1[j] = ioa;
            ioa++;
        } else {
            j -= NA;
            idxc1[i] = iob;
            idxb1[j] = iob;
            iob++;
        }
    }
    for(size_t i = 0; i < NA; i++) {
        if(conn[NC + i] < NC) continue;
        size_t j = conn[NC + i] - NC - NA;
        idxa1[i] = ii;
        idxb1[j] = ii;
        ii++;
    }

    //  Build matricized index ordering

    size_t iai, iao, ibi, ibo, ica, icb;
    if(idxa1[NA - 1] >= NC) {
        //  Last index in A is an inner index
        iai = NA; iao = N;
    } else {
        //  Last index in A is an outer index
        iai = K; iao = NA;
    }
    if(idxb1[NB - 1] >= NC) {
        //  Last index in B is an inner index
        ibi = NB; ibo = M;
    } else {
        //  Last index in B is an outer index
        ibi = K; ibo = NB;
    }
    if(idxc1[NC - 1] < N) {
        //  Last index in C comes from A
        ica = NC; icb = M;
    } else {
        //  Last index in C comes from B
        ica = N; icb = NC;
    }

    for(size_t i = 0; i < NA; i++) {
        if(idxa1[NA - i - 1] >= NC) {
            idxa2[iai - 1] = idxa1[NA - i - 1];
            iai--;
        } else {
            idxa2[iao - 1] = idxa1[NA - i - 1];
            iao--;
        }
    }
    for(size_t i = 0; i < NB; i++) {
        if(idxb1[NB - i - 1] >= NC) {
            idxb2[ibi - 1] = idxb1[NB - i - 1];
            ibi--;
        } else {
            idxb2[ibo - 1] = idxb1[NB - i - 1];
            ibo--;
        }
    }
    for(size_t i = 0; i < NC; i++) {
        if(idxc1[NC - i - 1] < N) {
            idxc2[ica - 1] = idxc1[NC - i - 1];
            ica--;
        } else {
            idxc2[icb - 1] = idxc1[NC - i - 1];
            icb--;
        }
    }

    bool lasta_i = (idxa2[NA - 1] >= NC);
    bool lastb_i = (idxb2[NB - 1] >= NC);
    bool lastc_a = (idxc2[NC - 1] < N);

    if(lastc_a) {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ji) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ji) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    } else {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ij) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[i];
                for(size_t i = 0; i < M; i++) idxb2[K + i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ij) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxc2[N + i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    }

    permutation_builder<NA> pba(idxa2, idxa1);
    permutation_builder<NB> pbb(idxb2, idxb1);
    permutation_builder<NC> pbc(idxc2, idxc1);
    perma.permute(pba.get_perm());
    permb.permute(pbb.get_perm());
    permc.permute(pbc.get_perm());
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
