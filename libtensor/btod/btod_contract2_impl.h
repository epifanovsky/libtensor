#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <memory>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/allocator.h>
#include "../core/mask.h"
#include "../symmetry/so_dirprod.h"
#include "../symmetry/so_reduce.h"
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/block_tensor/bto/bto_contract2_sym.h>
#include <libtensor/block_tensor/bto/bto_contract2_nzorb.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_set.h>
#include "btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2_clazz<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz =
    btod_contract2_clazz<N, M, K>::k_clazz;


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
    block_tensor_i<k_ordera, double> &bta,
    block_tensor_i<k_orderb, double> &btb) :

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

    clear_schedule(m_contr_sch);
}


template<size_t N, size_t M, size_t K>
const block_index_space<N + M> &btod_contract2<N, M, K>::get_bis() const {

    return m_symc.get_bisc();
}


template<size_t N, size_t M, size_t K>
const symmetry<N + M, double> &btod_contract2<N, M, K>::get_symmetry() const {

    return m_symc.get_symc();
}


template<size_t N, size_t M, size_t K>
const assignment_schedule<N + M, double>&
btod_contract2<N, M, K>::get_schedule() const {

    return m_sch;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::sync_on() {

    block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
    block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
    ctrla.req_sync_on();
    ctrlb.req_sync_on();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::sync_off() {

    block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
    block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
    ctrla.req_sync_off();
    ctrlb.req_sync_off();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<N + M, double> &btc) {

    block_tensor_ctrl<k_orderc, double> cc(btc);

    //  Prepare output tensor

    btod_set<k_orderc>().perform(btc);
    so_copy<k_orderc, double>(m_symc.get_symc()).perform(cc.req_symmetry());

    //  Compute

    perform(btc, 1.0);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<N + M, double> &btc,
    double d) {

    btod_contract2<N, M, K>::start_timer("perform");

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);

    //  Compute the number of blocks in A and B

    size_t nblka = 0, nblkb = 0;
    orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
    for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
        ioa != ola.end(); ++ioa) {
        if(!ca.req_is_zero_block(ola.get_index(ioa))) nblka++;
    }
    orbit_list<k_orderb, double> olb(cb.req_const_symmetry());
    for(typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
        iob != olb.end(); ++iob) {
        if(!cb.req_is_zero_block(olb.get_index(iob))) nblkb++;
    }

    if(nblka == 0 || nblkb == 0) {
        btod_contract2<N, M, K>::stop_timer("perform");
        return;
    }

    //  Number and size of batches in A and B

    size_t batsz = 128; // XXX: arbitrary batch size!
    size_t nbata, nbatb, batsza, batszb;
    nbata = (nblka + batsz - 1) / batsz;
    nbatb = (nblkb + batsz - 1) / batsz;
    batsza = nbata > 0 ? (nblka + nbata - 1) / nbata : 1;
    batszb = nbatb > 0 ? (nblkb + nbatb - 1) / nbatb : 1;

    //  Temporary partial A, B, and C

    block_tensor< k_ordera, double, allocator<double> > btat(m_bta.get_bis());
    block_tensor< k_orderb, double, allocator<double> > btbt(m_btb.get_bis());
    block_tensor< k_orderc, double, allocator<double> > btct1(m_symc.get_bisc()),
        btct2(m_symc.get_bisc());
    block_tensor_ctrl<k_ordera, double> cat(btat);
    block_tensor_ctrl<k_orderb, double> cbt(btbt);
    block_tensor_ctrl<k_orderc, double> cct1(btct1), cct2(btct2);
    so_copy<k_ordera, double>(ca.req_const_symmetry()).perform(
        cat.req_symmetry());
    so_copy<k_orderb, double>(cb.req_const_symmetry()).perform(
        cbt.req_symmetry());
    so_copy<k_orderc, double>(m_symc.get_symc()).perform(cct2.req_symmetry());

    //  List of canonical blocks in C
    //  Required for cases when batching breaks symmetry

    std::vector<size_t> blst;
    for(typename assignment_schedule<k_orderc, double>::iterator i =
        m_sch.begin(); i != m_sch.end(); ++i) {
        blst.push_back(m_sch.get_abs_index(i));
    }

    //  Batching

    dimensions<k_orderc> bidimsc = m_symc.get_bisc().get_block_index_dims();

    typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
    while(ioa != ola.end()) {

        btod_set<k_ordera>().perform(btat);
        btod_contract2<N, M, K>::start_timer("copy_a");
        size_t nba;
        for(nba = 0; ioa != ola.end() && nba < batsza; ++ioa) {
            const index<k_ordera> &ia = ola.get_index(ioa);
            if(ca.req_is_zero_block(ia)) continue;
            dense_tensor_i<k_ordera, double> &blka0 = ca.req_block(ia);
            dense_tensor_i<k_ordera, double> &blka = cat.req_block(ia);
            tod_copy<k_ordera>(blka0).perform(true, 1.0, blka);
            cat.ret_block(ia);
            ca.ret_block(ia);
            nba++;
        }
        btod_contract2<N, M, K>::stop_timer("copy_a");

        if(nba == 0) continue;

        typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
        while(iob != olb.end()) {

            btod_set<k_orderb>().perform(btbt);
            btod_contract2<N, M, K>::start_timer("copy_b");
            size_t nbb;
            for(nbb = 0; iob != olb.end() && nbb < batszb; ++iob) {
                const index<k_orderb> &ib = olb.get_index(iob);
                if(cb.req_is_zero_block(ib)) continue;
                dense_tensor_i<k_orderb, double> &blkb0 = cb.req_block(ib);
                dense_tensor_i<k_orderb, double> &blkb = cbt.req_block(ib);
                tod_copy<k_orderb>(blkb0).perform(true, 1.0, blkb);
                cbt.ret_block(ib);
                cb.ret_block(ib);
                nbb++;
            }
            btod_contract2<N, M, K>::stop_timer("copy_b");

            if(nbb == 0) continue;

            btod_set<k_orderc>().perform(btct1);
            so_copy<k_orderc, double>(m_symc.get_symc()).
                perform(cct1.req_symmetry());
            //  Calling this may break the symmetry of final result
            //  in some cases, e.g. self-contraction
            btod_contract2<N, M, K>(m_contr, btat, btbt).
                perform_inner(btct1, 1.0, blst);

            btod_contract2<N, M, K>::start_timer("copy_c");
            for(size_t i = 0; i < blst.size(); i++) {
                abs_index<k_orderc> aic(blst[i], bidimsc);
                if(!cct1.req_is_zero_block(aic.get_index())) {
                    bool zero = cct2.req_is_zero_block(aic.get_index());
                    dense_tensor_i<k_orderc, double> &blkc0 =
                        cct1.req_block(aic.get_index());
                    dense_tensor_i<k_orderc, double> &blkc =
                        cct2.req_block(aic.get_index());
                    tod_copy<k_orderc>(blkc0).perform(zero, 1.0, blkc);
                    cct1.ret_block(aic.get_index());
                    cct2.ret_block(aic.get_index());
                }
            }
            btod_contract2<N, M, K>::stop_timer("copy_c");
        }
    }

    btod_copy<k_orderc>(btct2).perform(btc, d);

    btod_contract2<N, M, K>::stop_timer("perform");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform_inner(
    block_tensor_i<N + M, double> &btc, double d,
    const std::vector<size_t> &blst) {

    additive_bto< N + M, bto_traits<double> >::perform(btc, d, blst);
}


/*
template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(dense_tensor_i<N + M, double> &blk,
    const index<N + M> &i) {

    static const char *method =
        "compute_block(dense_tensor_i<N + M, double>&, const index<N + M>&)";

    btod_contract2<N, M, K>::start_timer();

    try {

        block_tensor_ctrl<k_ordera, double> ca(m_bta);
        block_tensor_ctrl<k_orderb, double> cb(m_btb);

        abs_index<k_orderc> aic(i, m_bidimsc);
        typename schedule_t::iterator isch =
            m_contr_sch.find(aic.get_abs_index());
        if(isch == m_contr_sch.end()) {
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "i");
        }

        transf<k_orderc, double> trc0;
        contract_block(isch->second->first, aic.get_index(), ca, cb,
            blk, trc0, true, 1.0);

    } catch(...) {
        btod_contract2<N, M, K>::stop_timer();
        throw;
    }

    btod_contract2<N, M, K>::stop_timer();
}
*/

template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(bool zero,
    dense_tensor_i<N + M, double> &blk, const index<N + M> &i,
    const tensor_transf<N + M, double> &tr, const double &c) {

    static const char *method = "compute_block(bool, tensor_i<N + M, double>&, "
        "const index<N + M>&, const tensor_transf<N + M, double>&, "
        "const double&)";

    btod_contract2<N, M, K>::start_timer("compute_block");

    try {

        block_tensor_ctrl<k_ordera, double> ca(m_bta);
        block_tensor_ctrl<k_orderb, double> cb(m_btb);

        abs_index<k_orderc> aic(i, m_bidimsc);
        typename schedule_t::iterator isch =
            m_contr_sch.find(aic.get_abs_index());
        if(isch == m_contr_sch.end()) {
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "i");
        }

        contract_block(isch->second->first, aic.get_index(), ca, cb,
           blk, tr, zero, c);
    } catch(...) {
        btod_contract2<N, M, K>::stop_timer("compute_block");
        throw;
    }

    btod_contract2<N, M, K>::stop_timer("compute_block");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

    btod_contract2<N, M, K>::start_timer("make_schedule");
    btod_contract2<N, M, K>::start_timer("prepare_sch");

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);

    ca.req_sync_on();
    cb.req_sync_on();

    orbit_list<k_ordera, double> ola(ca.req_const_symmetry());

    std::vector<make_schedule_task*> tasklist;
    libutil::mutex sch_lock;

    typename orbit_list<k_ordera, double>::iterator ioa1 = ola.begin(),
        ioa2 = ola.begin();
    size_t n = 0, nmax = ola.get_size() / 64;
    if(nmax < 1024) nmax = 1024;
    for(; ioa2 != ola.end(); ioa2++, n++) {

        if(n == nmax) {
            make_schedule_task *t = new make_schedule_task(m_contr,
                    m_bta, m_btb, get_symmetry(), m_bidimsc, ola,
                    ioa1, ioa2, m_contr_sch, m_sch, sch_lock);
            tasklist.push_back(t);
            n = 0;
            ioa1 = ioa2;
        }
    }
    if(ioa1 != ola.end()) {
        make_schedule_task *t = new make_schedule_task(m_contr, m_bta,
                m_btb, get_symmetry(), m_bidimsc, ola, ioa1, ioa2,
                m_contr_sch, m_sch, sch_lock);
        tasklist.push_back(t);
    }

    btod_contract2<N, M, K>::stop_timer("prepare_sch");

    make_schedule_task_iterator ti(tasklist);
    make_schedule_task_observer to;
    libutil::thread_pool::submit(ti, to);

    for(size_t i = 0; i < tasklist.size(); i++) delete tasklist[i];

    ca.req_sync_off();
    cb.req_sync_off();

    for(typename schedule_t::iterator i = m_contr_sch.begin();
        i != m_contr_sch.end(); i++) {
        m_sch.insert(i->first);
    }

    btod_contract2<N, M, K>::stop_timer("make_schedule");

//    bto_contract2_nzorb<N, M, K, double> nzorb(m_contr, m_bta, m_btb,
//        m_symc.get_symc());
//    nzorb.build();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::clear_schedule(schedule_t &sch) {

    typename schedule_t::iterator isch = sch.begin();
    for(; isch != sch.end(); isch++) {
        delete isch->second;
        isch->second = 0;
    }
    sch.clear();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::make_schedule_task::make_schedule_task(
    const contraction2<N, M, K> &contr,
    block_tensor_i<k_ordera, double> &bta,
    block_tensor_i<k_orderb, double> &btb,
    const symmetry<k_orderc, double> &symc,
    const dimensions<k_orderc> &bidimsc,
    const orbit_list<k_ordera, double> &ola,
    const typename orbit_list<k_ordera, double>::iterator &ioa1,
    const typename orbit_list<k_ordera, double>::iterator &ioa2,
    schedule_t &contr_sch, assignment_schedule<k_orderc, double> &sch,
    libutil::mutex &sch_lock) :

    m_contr(contr),
    m_ca(bta), m_bidimsa(bta.get_bis().get_block_index_dims()),
    m_cb(btb), m_bidimsb(btb.get_bis().get_block_index_dims()),
    m_symc(symc), m_bidimsc(bidimsc),
    m_ola(ola), m_ioa1(ioa1), m_ioa2(ioa2),
    m_contr_sch(contr_sch), m_sch(sch), m_sch_lock(sch_lock) {

}


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::make_schedule_task::k_clazz =
    "btod_contract2<N, M, K>::make_schedule_task";


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::perform() {

    make_schedule_task::start_timer("local");

    orbit_list<k_orderc, double> olc(m_symc);

    for(typename orbit_list<k_ordera, double>::iterator ioa = m_ioa1;
        ioa != m_ioa2; ioa++) {

        orbit<k_ordera, double> oa(m_ca.req_const_symmetry(),
                                   m_ola.get_index(ioa));
        if(!oa.is_allowed()) continue;
        if(m_ca.req_is_zero_block(m_ola.get_index(ioa))) continue;

        abs_index<k_ordera> acia(m_ola.get_index(ioa), m_bidimsa);
        for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
            ia != oa.end(); ia++) {

            abs_index<k_ordera> aia(oa.get_abs_index(ia),
                                    m_bidimsa);
            make_schedule_a(olc, aia, acia, oa.get_transf(ia));
        }
    }

    make_schedule_task::stop_timer("local");

    make_schedule_task::start_timer("merge");
    merge_schedule();
    make_schedule_task::stop_timer("merge");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::make_schedule_a(
    const orbit_list<k_orderc, double> &olc,
    const abs_index<k_ordera> &aia, const abs_index<k_ordera> &acia,
    const tensor_transf<k_ordera, double> &tra) {

    const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
    const index<k_ordera> &ia = aia.get_index();

    sequence<M, size_t> useqb(0), useqc(0); // Maps uncontracted indexes
    index<M> iu1, iu2;
    for(size_t i = 0, j = 0; i < k_orderc; i++) {
        if(conn[i] >= k_orderc + k_ordera) {
            useqb[j] = conn[i] - k_orderc - k_ordera;
            useqc[j] = i;
            j++;
        }
    }
    for(size_t i = 0; i < M; i++) iu2[i] = m_bidimsb[useqb[i]] - 1;
    // Uncontracted indexes from B
    dimensions<M> bidimsu(index_range<M>(iu1, iu2));
    abs_index<M> aiu(bidimsu);
    do {
        const index<M> &iu = aiu.get_index();

        // Construct the index in C
        index<k_orderc> ic;
        for(size_t i = 0, j = 0; i < k_orderc; i++) {
            if(conn[i] < k_orderc + k_ordera) {
                ic[i] = ia[conn[i] - k_orderc];
            } else {
                ic[i] = iu[j++];
            }
        }

        // Skip non-canonical indexes in C
        abs_index<k_orderc> aic(ic, m_bidimsc);
        if(!olc.contains(aic.get_abs_index())) continue;

        // Construct the index in B
        index<k_orderb> ib;
        for(size_t i = 0; i < k_orderb; i++) {
            register size_t k = conn[k_orderc + k_ordera + i];
            if(k < k_orderc) ib[i] = ic[k];
            else ib[i] = ia[k - k_orderc];
        }
        make_schedule_b(acia, tra, ib, aic);
    } while(aiu.inc());
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::make_schedule_b(
    const abs_index<k_ordera> &acia, const tensor_transf<k_ordera, double> &tra,
    const index<k_orderb> &ib, const abs_index<k_orderc> &acic) {

    orbit<k_orderb, double> ob(m_cb.req_const_symmetry(), ib);
    if(!ob.is_allowed()) return;

    abs_index<k_orderb> acib(ob.get_abs_canonical_index(), m_bidimsb);
    if(m_cb.req_is_zero_block(acib.get_index())) return;

    const tensor_transf<k_orderb, double> &trb = ob.get_transf(ib);
    block_contr_t bc(acia.get_abs_index(), acib.get_abs_index(),
                     tra.get_scalar_tr().get_coeff() *
                     trb.get_scalar_tr().get_coeff(),
                     permutation<N + K>(tra.get_perm(), true),
                     permutation<M + K>(trb.get_perm(), true));
    schedule_block_contraction(acic, bc);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::schedule_block_contraction(
    const abs_index<k_orderc> &acic, const block_contr_t &bc) {

    typename schedule_t::iterator isch;
    block_contr_list_pair_t *lstpair = 0;

    std::pair<typename schedule_t::iterator, bool> r =
        m_contr_sch_local.insert(
            std::pair<size_t, block_contr_list_pair_t*>(
                acic.get_abs_index(), lstpair));
    // Check whether this is the first contraction for this block in C
    if(r.second) {
        lstpair = new block_contr_list_pair_t;
        lstpair->first.push_back(bc);
        lstpair->second = false;
        r.first->second = lstpair;
        return;
    } else {
        isch = r.first;
    }

    lstpair = isch->second;
    merge_node(bc, lstpair->first, lstpair->first.begin());
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::merge_schedule() {

    libutil::auto_lock<libutil::mutex> lock(m_sch_lock);

    for(typename schedule_t::iterator isrc = m_contr_sch_local.begin();
        isrc != m_contr_sch_local.end(); isrc++) {

        std::pair<typename schedule_t::iterator, bool> rdst =
            m_contr_sch.insert(*isrc);
        if(!rdst.second) {
            typename schedule_t::iterator idst = rdst.first;
            merge_lists(isrc->second->first, idst->second->first);
            delete isrc->second;
        }
        isrc->second = 0;
    }
    m_contr_sch_local.clear();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::merge_lists(
    const block_contr_list_t &src, block_contr_list_t &dst) {

    typename block_contr_list_t::const_iterator isrc = src.begin();
    typename block_contr_list_t::iterator idst = dst.begin();
    for(; isrc != src.end(); isrc++) {
        idst = merge_node(*isrc, dst, idst);
    }
}


template<size_t N, size_t M, size_t K>
typename btod_contract2<N, M, K>::block_contr_list_iterator_t
btod_contract2<N, M, K>::make_schedule_task::merge_node(
    const block_contr_t &bc, block_contr_list_t &lst,
    const block_contr_list_iterator_t &begin) {

    typename block_contr_list_t::iterator ilst = begin;

    while(ilst != lst.end() && ilst->m_absidxa < bc.m_absidxa) ilst++;
    while(ilst != lst.end() && ilst->m_absidxa == bc.m_absidxa &&
          ilst->m_absidxb < bc.m_absidxb) ilst++;

    // If similar contractions are found, try to combine with them
    bool done = false;
    while(!done && ilst != lst.end() && ilst->m_absidxa == bc.m_absidxa &&
          ilst->m_absidxb == bc.m_absidxb) {

        if(ilst->m_perma.equals(bc.m_perma) &&
           ilst->m_permb.equals(bc.m_permb)) {
            ilst->m_c += bc.m_c;
            if(ilst->m_c == 0.0) {
                lst.erase(ilst);
                ilst = lst.end();
            }
            done = true;
        } else {
            ilst++;
        }
    }

    // If similar contractions are not found, simply add bc
    if(!done) lst.insert(ilst, bc);

    return ilst;
}


template<size_t N, size_t M, size_t K>
bool btod_contract2<N, M, K>::make_schedule_task_iterator::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, size_t M, size_t K>
libutil::task_i *btod_contract2<N, M, K>::make_schedule_task_iterator::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
    block_contr_list_t &lst, const index<k_orderc> &idxc,
    block_tensor_ctrl<k_ordera, double> &ca,
    block_tensor_ctrl<k_orderb, double> &cb,
    dense_tensor_i<k_orderc, double> &tc,
    const tensor_transf<k_orderc, double> &trc,
    bool zero, double c) {

    std::list< index<k_ordera> > blksa;
    std::list< index<k_orderb> > blksb;

    std::auto_ptr< tod_contract2<N, M, K> > op;

    for(typename block_contr_list_t::iterator ilst = lst.begin();
        ilst != lst.end(); ilst++) {

        abs_index<k_ordera> aia(ilst->m_absidxa, m_bidimsa);
        abs_index<k_orderb> aib(ilst->m_absidxb, m_bidimsb);
        const index<k_ordera> &ia = aia.get_index();
        const index<k_orderb> &ib = aib.get_index();

        bool zeroa = ca.req_is_zero_block(ia);
        bool zerob = cb.req_is_zero_block(ib);
        if(zeroa || zerob) continue;

        dense_tensor_i<k_ordera, double> &blka = ca.req_block(ia);
        dense_tensor_i<k_orderb, double> &blkb = cb.req_block(ib);
        blksa.push_back(ia);
        blksb.push_back(ib);

        contraction2<N, M, K> contr(m_contr);
        contr.permute_a(ilst->m_perma);
        contr.permute_b(ilst->m_permb);
        contr.permute_c(trc.get_perm());

        double kc = ilst->m_c * trc.get_scalar_tr().get_coeff();
        if(op.get() == 0) {
            op = std::auto_ptr< tod_contract2<N, M, K> >(
                new tod_contract2<N, M, K>(contr, blka, blkb, kc));
        } else {
            op->add_args(contr, blka, blkb, kc);
        }
    }

    op->perform(zero, c, tc);

    for(typename std::list< index<k_ordera> >::const_iterator i =
            blksa.begin(); i != blksa.end(); i++) ca.ret_block(*i);
    for(typename std::list< index<k_orderb> >::const_iterator i =
            blksb.begin(); i != blksb.end(); i++) cb.ret_block(*i);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
