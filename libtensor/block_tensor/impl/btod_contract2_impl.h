#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <map>
#include <memory>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_clst_impl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_nzorb_impl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_sym_impl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_impl.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include <libtensor/block_tensor/impl/bto_stream_adapter.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2_clazz<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz =
    btod_contract2_clazz<N, M, K>::k_clazz;


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, double> &bta,
    block_tensor_rd_i<NB, double> &btb) :

    m_contr(contr), m_bta(bta), m_btb(btb),
    m_symc(contr, bta, btb),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bisc().get_block_index_dims()),
    m_sch(m_bidimsc) {

    make_schedule();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {

}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(bto_stream_i<NC, btod_traits> &out) {

    typedef block_tensor_i_traits<double> bti_traits;
    typedef gen_bto_copy< NA, btod_traits, btod_contract2<N, M, K> >
        gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, btod_traits, btod_contract2<N, M, K> >
        gen_bto_copy_b_type;
    typedef gen_bto_copy< NC, btod_traits, btod_contract2<N, M, K> >
        gen_bto_copy_c_type;

    btod_contract2<N, M, K>::start_timer();

    try {

        out.open();

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        //  Compute the number of blocks in A and B

        size_t nblka = 0, nblkb = 0, nblkc = 0;
        orbit_list<NA, double> ola(ca.req_const_symmetry());
        for(typename orbit_list<NA, double>::iterator ioa = ola.begin();
            ioa != ola.end(); ++ioa) {
            if(!ca.req_is_zero_block(ola.get_index(ioa))) nblka++;
        }
        orbit_list<NB, double> olb(cb.req_const_symmetry());
        for(typename orbit_list<NB, double>::iterator iob = olb.begin();
            iob != olb.end(); ++iob) {
            if(!cb.req_is_zero_block(olb.get_index(iob))) nblkb++;
        }
        for(typename assignment_schedule<NC, double>::iterator i =
            m_sch.begin(); i != m_sch.end(); ++i) {
            nblkc++;
        }

        //  Quit if either one of the arguments is zero

        if(nblka == 0 || nblkb == 0) {
            out.close();
            btod_contract2<N, M, K>::stop_timer();
            return;
        }

        //  Number and size of batches in A, B and C

        size_t batsz = 4096; // XXX: arbitrary batch size!
        size_t nbata, nbatb, nbatc, batsza, batszb, batszc;
        nbata = (nblka + batsz - 1) / batsz;
        nbatb = (nblkb + batsz - 1) / batsz;
        nbatc = (nblkc + batsz - 1) / batsz;
        batsza = nbata > 0 ? (nblka + nbata - 1) / nbata : 1;
        batszb = nbatb > 0 ? (nblkb + nbatb - 1) / nbatb : 1;
        batszc = nbatc > 0 ? (nblkc + nbatc - 1) / nbatc : 1;

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

        symmetry<NA, double> symat(bisat);
        symmetry<NB, double> symbt(bisbt);
        symmetry<NC, double> symct(bisct);
        so_permute<NA, double>(ca.req_const_symmetry(), perma).perform(symat);
        so_permute<NB, double>(cb.req_const_symmetry(), permb).perform(symbt);
        so_permute<NC, double>(m_symc.get_symc(), permc).perform(symct);

        //  Temporary partial A, B, and C

        block_tensor< NA, double, allocator<double> > btat(bisat);
        block_tensor< NB, double, allocator<double> > btbt(bisbt);
        block_tensor< NC, double, allocator<double> > btct(bisct);
        block_tensor_ctrl<NA, double> cat(btat);
        block_tensor_ctrl<NB, double> cbt(btbt);
        block_tensor_ctrl<NC, double> cct(btct);

        //  Batching loops

        dimensions<NC> bidimsc = m_symc.get_bisc().get_block_index_dims();
        dimensions<NC> bidimsct = bisct.get_block_index_dims();
        std::vector<size_t> batcha, batchb, batchc1, batchc2;
        batcha.reserve(batsza);
        batchb.reserve(batszb);
        batchc1.reserve(batszc);
        batchc2.reserve(batszc);

        typename orbit_list<NA, double>::iterator ioa = ola.begin();
        bool first_batch_a = true;
        while(ioa != ola.end()) {

            batcha.clear();
            if(perma.is_identity()) {
                for(; ioa != ola.end() && batcha.size() < batsza; ++ioa) {
                    const index<NA> &ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    batcha.push_back(ola.get_abs_index(ioa));
                }
            } else {
                for(; ioa != ola.end() && batcha.size() < batsza; ++ioa) {
                    index<NA> ia = ola.get_index(ioa);
                    if(ca.req_is_zero_block(ia)) continue;
                    ia.permute(perma);
                    orbit<NA, double> oat(symat, ia, false);
                    batcha.push_back(oat.get_acindex());
                }
            }

            //  If A need not be permuted and fits in one batch entirely,
            //  do not make a copy; use the original tensor
            bool use_orig_a = (first_batch_a && ioa == ola.end() &&
                perma.is_identity());
            first_batch_a = false;

            if(!use_orig_a) {
                btod_contract2<N, M, K>::start_timer("copy_a");
                tensor_transf<NA, double> tra(perma);
                bto_aux_copy<NA, btod_traits> cpaout(symat, btat);
                bto_stream_adapter<NA, btod_traits> cpaout1(cpaout);
                gen_bto_copy_a_type(m_bta, tra).perform(batcha, cpaout1);
                btod_contract2<N, M, K>::stop_timer("copy_a");
            }

            block_tensor_rd_i<NA, double> &bta = use_orig_a ? m_bta : btat;

            if(batcha.size() == 0) continue;

            typename orbit_list<NB, double>::iterator iob = olb.begin();
            bool first_batch_b = true;
            while(iob != olb.end()) {

                batchb.clear();
                if(permb.is_identity()) {
                    for(; iob != olb.end() && batchb.size() < batszb; ++iob) {
                        const index<NB> &ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        batchb.push_back(olb.get_abs_index(iob));
                    }
                } else {
                    for(; iob != olb.end() && batchb.size() < batszb; ++iob) {
                        index<NB> ib = olb.get_index(iob);
                        if(cb.req_is_zero_block(ib)) continue;
                        ib.permute(permb);
                        orbit<NB, double> obt(symbt, ib, false);
                        batchb.push_back(obt.get_acindex());
                    }
                }

                bool use_orig_b = (first_batch_b && iob == olb.end() &&
                    permb.is_identity());
                first_batch_b = false;

                if(!use_orig_b) {
                    btod_contract2<N, M, K>::start_timer("copy_b");
                    tensor_transf<NB, double> trb(permb);
                    bto_aux_copy<NB, btod_traits> cpbout(symbt, btbt);
                    bto_stream_adapter<NB, btod_traits> cpbout1(cpbout);
                    gen_bto_copy_b_type(m_btb, trb).perform(batchb, cpbout1);
                    btod_contract2<N, M, K>::stop_timer("copy_b");
                }

                block_tensor_rd_i<NB, double> &btb = use_orig_b ? m_btb : btbt;

                if(batchb.size() == 0) continue;

                typename assignment_schedule<NC, double>::iterator ibc =
                    m_sch.begin();
                while(ibc != m_sch.end()) {

                    batchc1.clear();
                    batchc2.clear();

                    for(; ibc != m_sch.end() && batchc1.size() < batszc;
                        ++ibc) {
                        index<NC> ic;
                        abs_index<NC>::get_index(m_sch.get_abs_index(ibc),
                            bidimsc, ic);
                        ic.permute(permc);
                        orbit<NC, double> oct(symct, ic, false);
                        batchc1.push_back(oct.get_acindex());
                    }
                    if(batchc1.size() == 0) continue;

                    //  Calling this may break the symmetry of final result
                    //  in some cases, e.g. self-contraction
                    gen_bto_contract2<N, M, K, btod_traits, btod_contract2> bto(
                        contr, bta, btb);
                    bto_aux_copy<NC, btod_traits> ctcout(symct, btct);
                    bto_stream_adapter<NC, btod_traits> ctcout1(ctcout);
                    bto.perform(batchc1, ctcout1);

                    btod_contract2<N, M, K>::start_timer("copy_c");
                    for(size_t i = 0; i < batchc1.size(); i++) {
                        index<NC> ic;
                        abs_index<NC>::get_index(batchc1[i], bidimsct, ic);
                        if(!cct.req_is_zero_block(ic)) {
                            ic.permute(permcinv);
                            orbit<NC, double> oc(m_symc.get_symc(), ic, false);
                            batchc2.push_back(oc.get_acindex());
                        }
                    }
                    tensor_transf<NC, double> trc(permcinv);
                    bto_stream_adapter<NC, btod_traits> cpcout1(out);
                    gen_bto_copy_c_type(btct, trc).perform(batchc2, cpcout1);
                    btod_contract2<N, M, K>::stop_timer("copy_c");
                }
            }
        }

        out.close();

    } catch(...) {
        btod_contract2<N, M, K>::stop_timer();
        throw;
    }

    btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<NC, double> &btc) {

    bto_aux_copy<NC, btod_traits> out(m_symc.get_symc(), btc);
    perform(out);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(
    block_tensor_i<NC, double> &btc,
    const double &d) {

    block_tensor_ctrl<NC, double> cc(btc);
    addition_schedule<NC, btod_traits> asch(m_symc.get_symc(),
        cc.req_const_symmetry());
    asch.build(m_sch, cc);

    bto_aux_add<NC, btod_traits> out(m_symc.get_symc(), asch, btc, d);
    perform(out);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(
    bool zero,
    dense_tensor_i<NC, double> &blk,
    const index<NC> &i,
    const tensor_transf<NC, double> &tr,
    const double &c) {

    typedef block_tensor_i_traits<double> bti_traits;

    btod_contract2::start_timer("compute_block");

    try {

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        const symmetry<NA, double> &syma = ca.req_const_symmetry();
        const symmetry<NB, double> &symb = cb.req_const_symmetry();

        orbit_list<NA, double> ola(syma);
        orbit_list<NB, double> olb(symb);

        abs_index<NC> aic(i, m_bidimsc);
        contract_block(m_bta, ola, m_btb, olb, aic.get_index(), blk, tr, zero,
            c);

    } catch(...) {
        btod_contract2::stop_timer("compute_block");
        throw;
    }

    btod_contract2::stop_timer("compute_block");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

    btod_contract2::start_timer("make_schedule");

    gen_bto_contract2_nzorb<N, M, K, btod_traits, btod_contract2> nzorb(m_contr,
        m_bta, m_btb, m_symc.get_symc());
    nzorb.build();
    for(typename std::vector<size_t>::const_iterator i =
        nzorb.get_blst().begin(); i != nzorb.get_blst().end(); ++i) {
        m_sch.insert(*i);
    }

    btod_contract2::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::align(
    const sequence<2 * (N + M + K), size_t> &conn,
    permutation<N + K> &perma, permutation<M + K> &permb,
    permutation<N + M> &permc) {

    //  This algorithm reorders indexes in A, B, C so that the whole contraction
    //  can be done in a single matrix multiplication.
    //  Returned permutations perma, permb, permc need to be applied to
    //  the indexes of A, B, and C to get the matricized form.

    //  Numbering scheme:
    //  0     .. N - 1         -- outer indexes from A
    //  N     .. N + M - 1     -- outer indexes from B
    //  N + M .. N + M + K - 1 -- inner indexes

    size_t ioa = 0, iob = N, ii = N + M;

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
    if(idxa1[NA - 1] >= N + M) {
        //  Last index in A is an inner index
        iai = N + K; iao = N;
    } else {
        //  Last index in A is an outer index
        iai = K; iao = N + K;
    }
    if(idxb1[NB - 1] >= N + M) {
        //  Last index in B is an inner index
        ibi = M + K; ibo = M;
    } else {
        //  Last index in B is an outer index
        ibi = K; ibo = M + K;
    }
    if(idxc1[NC - 1] < N) {
        //  Last index in C comes from A
        ica = N + M; icb = M;
    } else {
        //  Last index in C comes from B
        ica = N; icb = N + M;
    }

    for(size_t i = 0; i < NA; i++) {
        if(idxa1[NA - i - 1] >= N + M) {
            idxa2[iai - 1] = idxa1[NA - i - 1];
            iai--;
        } else {
            idxa2[iao - 1] = idxa1[NA - i - 1];
            iao--;
        }
    }
    for(size_t i = 0; i < NB; i++) {
        if(idxb1[NB - i - 1] >= N + M) {
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

    bool lasta_i = (idxa2[NA - 1] >= N + M);
    bool lastb_i = (idxb2[NB - 1] >= N + M);
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


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
    block_tensor_rd_i<N + K, double> &bta,
    const orbit_list<N + K, double> &ola,
    block_tensor_rd_i<M + K, double> &btb,
    const orbit_list<M + K, double> &olb,
    const index<NC> &idxc,
    dense_tensor_i<NC, double> &tc,
    const tensor_transf<NC, double> &trc,
    bool zero, double c) {

    typedef block_tensor_i_traits<double> bti_traits;
    typedef typename gen_bto_contract2_clst<N, M, K, btod_traits>::contr_list contr_list;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    //  Prepare contraction list
    btod_contract2<N, M, K>::start_timer("contract_block::clst");
    gen_bto_contract2_clst<N, M, K, btod_traits> clstop(m_contr, bta, btb, ola, olb,
        m_bidimsa, m_bidimsb, m_bidimsc, idxc);
    clstop.build_list(false); // Build full contraction list
    const contr_list &clst = clstop.get_clst();
    btod_contract2<N, M, K>::stop_timer("contract_block::clst");

    //  Keep track of checked out blocks
    typedef std::map<size_t, dense_tensor_i<NA, double>*> coba_map;
    typedef std::map<size_t, dense_tensor_i<NB, double>*> cobb_map;
    coba_map coba;
    cobb_map cobb;

    //  Tensor contraction operation
    std::auto_ptr< tod_contract2<N, M, K> > op;

    //  Go through the contraction list and prepare the contraction
    for(typename contr_list::const_iterator i = clst.begin();
        i != clst.end(); ++i) {

        index<NA> ia;
        index<NB> ib;
        abs_index<NA>::get_index(i->aia, m_bidimsa, ia);
        abs_index<NB>::get_index(i->aib, m_bidimsb, ib);

        bool zeroa = ca.req_is_zero_block(ia);
        bool zerob = cb.req_is_zero_block(ib);
        if(zeroa || zerob) continue;

        if(coba.find(i->aia) == coba.end()) {
            dense_tensor_i<NA, double> &ta = ca.req_const_block(ia);
            coba[i->aia] = &ta;
        }
        if(cobb.find(i->aib) == cobb.end()) {
            dense_tensor_i<NB, double> &tb = cb.req_const_block(ib);
            cobb[i->aib] = &tb;
        }
        dense_tensor_i<NA, double> &ta = *coba[i->aia];
        dense_tensor_i<NB, double> &tb = *cobb[i->aib];

        tensor_transf<NA, double> trainv(i->tra);
        trainv.invert();
        tensor_transf<NB, double> trbinv(i->trb);
        trbinv.invert();

        contraction2<N, M, K> contr(m_contr);
        contr.permute_a(trainv.get_perm());
        contr.permute_b(trbinv.get_perm());
        contr.permute_c(trc.get_perm());

        double kc = trainv.get_scalar_tr().get_coeff() *
            trbinv.get_scalar_tr().get_coeff() *
            trc.get_scalar_tr().get_coeff();

        if(op.get() == 0) {
            op = std::auto_ptr< tod_contract2<N, M, K> >(
                new tod_contract2<N, M, K>(contr, ta, tb, kc));
        } else {
            op->add_args(contr, ta, tb, kc);
        }
    }

    //  Execute the contraction
    if(op.get() == 0) {
        if(zero) tod_set<NC>().perform(tc);
    } else {
        op->perform(zero, c, tc);
    }

    //  Return input blocks
    for(typename coba_map::iterator i = coba.begin(); i != coba.end(); ++i) {
        index<NA> ia;
        abs_index<NA>::get_index(i->first, m_bidimsa, ia);
        ca.ret_const_block(ia);
    }
    for(typename cobb_map::iterator i = cobb.begin(); i != cobb.end(); ++i) {
        index<NB> ib;
        abs_index<NB>::get_index(i->first, m_bidimsb, ib);
        cb.ret_const_block(ib);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
