#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <map>
#include <memory>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/mask.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/block_tensor/bto/bto_contract2_clst.h>
#include <libtensor/block_tensor/bto/bto_contract2_nzorb.h>
#include <libtensor/block_tensor/bto/bto_contract2_sym.h>
#include <libtensor/block_tensor/btod/btod_copy.h>
#include <libtensor/block_tensor/btod/btod_set.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2_clazz<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz =
    btod_contract2_clazz<N, M, K>::k_clazz;


template<size_t N, typename T>
class btod_contract2_copy_task : public libutil::task_i {
private:
    block_tensor_i<N, T> &m_bta;
    block_tensor_i<N, T> &m_btb;
    index<N> m_idx;
    bool m_zero;

public:
    btod_contract2_copy_task(
        block_tensor_i<N, T> &bta,
        block_tensor_i<N, T> &btb,
        const index<N> &idx,
        bool zero);

    virtual ~btod_contract2_copy_task() { }
    virtual void perform();

};


template<size_t N, typename T>
class btod_contract2_copy_task_iterator : public libutil::task_iterator_i {
private:
    block_tensor_i<N, T> &m_bta;
    block_tensor_i<N, T> &m_btb;
    bool m_zero;
    const std::vector<size_t> &m_bi;
    typename std::vector<size_t>::const_iterator m_i;

public:
    btod_contract2_copy_task_iterator(
        block_tensor_i<N, T> &bta,
        block_tensor_i<N, T> &btb,
        bool zero,
        const std::vector<size_t> &bi);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename T>
class btod_contract2_copy_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


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

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);
    ca.req_sync_on();
    cb.req_sync_on();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::sync_off() {

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);
    ca.req_sync_off();
    cb.req_sync_off();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<N + M, double> &btc) {

    btod_contract2<N, M, K>::start_timer();

    try {

    btod_set<k_orderc>().perform(btc);

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);
    block_tensor_ctrl<k_orderc, double> cc(btc);

    so_copy<k_orderc, double>(m_symc.get_symc()).perform(cc.req_symmetry());

    //  Compute the number of blocks in A and B

    size_t nblka = 0, nblkb = 0, nblkc = 0;
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
    for(typename assignment_schedule<k_orderc, double>::iterator i =
        m_sch.begin(); i != m_sch.end(); ++i) {
        nblkc++;
    }

    if(nblka == 0 || nblkb == 0) {
        btod_contract2<N, M, K>::stop_timer("perform");
        return;
    }

    //  Number and size of batches in A, B and C

    size_t batsz = 128; // XXX: arbitrary batch size!
    size_t nbata, nbatb, nbatc, batsza, batszb, batszc;
    nbata = (nblka + batsz - 1) / batsz;
    nbatb = (nblkb + batsz - 1) / batsz;
    nbatc = (nblkc + batsz - 1) / batsz;
    batsza = nbata > 0 ? (nblka + nbata - 1) / nbata : 1;
    batszb = nbatb > 0 ? (nblkb + nbatb - 1) / nbatb : 1;
    batszc = nbatc > 0 ? (nblkc + nbatc - 1) / nbatc : 1;

    //  Temporary partial A, B, and C

    block_tensor< k_ordera, double, allocator<double> > btat(m_bta.get_bis());
    block_tensor< k_orderb, double, allocator<double> > btbt(m_btb.get_bis());
    block_tensor< k_orderc, double, allocator<double> > btct(m_symc.get_bisc());
    block_tensor_ctrl<k_ordera, double> cat(btat);
    block_tensor_ctrl<k_orderb, double> cbt(btbt);
    block_tensor_ctrl<k_orderc, double> cct(btct);
    so_copy<k_ordera, double>(ca.req_const_symmetry()).perform(
        cat.req_symmetry());
    so_copy<k_orderb, double>(cb.req_const_symmetry()).perform(
        cbt.req_symmetry());

    //  Batching loops

    dimensions<k_orderc> bidimsc = m_symc.get_bisc().get_block_index_dims();
    std::vector<size_t> batcha, batchb, batchc1, batchc2;
    batcha.reserve(batsza);
    batchb.reserve(batszb);
    batchc1.reserve(batszc);
    batchc2.reserve(batszc);

    typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
    while(ioa != ola.end()) {

        btod_set<k_ordera>().perform(btat);

        btod_contract2<N, M, K>::start_timer("copy_a");
        batcha.clear();
        for(; ioa != ola.end() && batcha.size() < batsza; ++ioa) {
            const index<k_ordera> &ia = ola.get_index(ioa);
            if(ca.req_is_zero_block(ia)) continue;
            batcha.push_back(ola.get_abs_index(ioa));
        }
        if(batcha.size() > 0) {
            ca.req_sync_on();
            cat.req_sync_on();
            btod_contract2_copy_task_iterator<N + K, double> ti(m_bta, btat,
                true, batcha);
            btod_contract2_copy_task_observer<N + K, double> to;
            libutil::thread_pool::submit(ti, to);
            cat.req_sync_off();
            ca.req_sync_off();
        }
        btod_contract2<N, M, K>::stop_timer("copy_a");

        if(batcha.size() == 0) continue;

        typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
        while(iob != olb.end()) {

            btod_set<k_orderb>().perform(btbt);

            btod_contract2<N, M, K>::start_timer("copy_b");
            batchb.clear();
            for(; iob != olb.end() && batchb.size() < batszb; ++iob) {
                const index<k_orderb> &ib = olb.get_index(iob);
                if(cb.req_is_zero_block(ib)) continue;
                batchb.push_back(olb.get_abs_index(iob));
            }
            if(batchb.size() > 0) {
                cb.req_sync_on();
                cbt.req_sync_on();
                btod_contract2_copy_task_iterator<M + K, double> ti(m_btb, btbt,
                    true, batchb);
                btod_contract2_copy_task_observer<M + K, double> to;
                libutil::thread_pool::submit(ti, to);
                cbt.req_sync_off();
                cb.req_sync_off();
            }
            btod_contract2<N, M, K>::stop_timer("copy_b");

            if(batchb.size() == 0) continue;

            typename assignment_schedule<k_orderc, double>::iterator ibc =
                m_sch.begin();
            while(ibc != m_sch.end()) {

                btod_set<k_orderc>().perform(btct);
                so_copy<k_orderc, double>(m_symc.get_symc()).
                    perform(cct.req_symmetry());

                batchc1.clear();
                batchc2.clear();

                for(; ibc != m_sch.end() && batchc1.size() < batszc; ++ibc) {
                    batchc1.push_back(m_sch.get_abs_index(ibc));
                }
                if(batchc1.size() == 0) continue;

                //  Calling this may break the symmetry of final result
                //  in some cases, e.g. self-contraction
                btod_contract2<N, M, K>(m_contr, btat, btbt).
                    perform_inner(btct, 1.0, batchc1);

                btod_contract2<N, M, K>::start_timer("copy_c");
                for(size_t i = 0; i < batchc1.size(); i++) {
                    index<k_orderc> ic;
                    abs_index<k_orderc>::get_index(batchc1[i], bidimsc, ic);
                    if(!cct.req_is_zero_block(ic)) {
                        batchc2.push_back(batchc1[i]);
                    }
                }
                if(batchc2.size() > 0) {
                    cct.req_sync_on();
                    cc.req_sync_on();
                    btod_contract2_copy_task_iterator<N + M, double> ti(btct,
                        btc, false, batchc2);
                    btod_contract2_copy_task_observer<N + M, double> to;
                    libutil::thread_pool::submit(ti, to);
                    cc.req_sync_off();
                    cct.req_sync_off();
                }
                btod_contract2<N, M, K>::stop_timer("copy_c");
            }
        }
    }

    } catch(...) {
        btod_contract2<N, M, K>::stop_timer();
        throw;
    }

    btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<N + M, double> &btc,
    double d) {

    block_tensor< k_orderc, double, allocator<double> > btct(m_symc.get_bisc());
    perform(btct);
    btod_copy<k_orderc>(btct).perform(btc, d);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform_inner(
    block_tensor_i<N + M, double> &btc, double d,
    const std::vector<size_t> &blst) {

    additive_bto< N + M, bto_traits<double> >::perform(btc, d, blst);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(bool zero,
    dense_tensor_i<N + M, double> &blk, const index<N + M> &i,
    const tensor_transf<N + M, double> &tr, const double &c) {

    static const char *method = "compute_block(bool, tensor_i<N + M, double>&, "
        "const index<N + M>&, const tensor_transf<N + M, double>&, "
        "const double&)";

    btod_contract2<N, M, K>::start_timer("compute_block");

    try {

        abs_index<k_orderc> aic(i, m_bidimsc);
        contract_block(aic.get_index(), blk, tr, zero, c);

    } catch(...) {
        btod_contract2<N, M, K>::stop_timer("compute_block");
        throw;
    }

    btod_contract2<N, M, K>::stop_timer("compute_block");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

    btod_contract2<N, M, K>::start_timer("make_schedule");

    bto_contract2_nzorb<N, M, K, double> nzorb(m_contr, m_bta, m_btb,
        m_symc.get_symc());
    nzorb.build();
    for(typename std::vector<size_t>::const_iterator i =
        nzorb.get_blst().begin(); i != nzorb.get_blst().end(); ++i) {
        m_sch.insert(*i);
    }

    btod_contract2<N, M, K>::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
    const index<k_orderc> &idxc,
    dense_tensor_i<k_orderc, double> &tc,
    const tensor_transf<k_orderc, double> &trc,
    bool zero, double c) {

    typedef typename bto_contract2_clst<N, M, K, double>::contr_list contr_list;

    block_tensor_ctrl<k_ordera, double> ca(m_bta);
    block_tensor_ctrl<k_orderb, double> cb(m_btb);

    //  Prepare contraction list
    btod_contract2<N, M, K>::start_timer("contract_block::clst");
    bto_contract2_clst<N, M, K, double> clstop(m_contr, m_bta, m_btb, m_bidimsa,
        m_bidimsb, m_bidimsc, idxc);
    clstop.build_list(false); // Build full contraction list
    const contr_list &clst = clstop.get_clst();
    btod_contract2<N, M, K>::stop_timer("contract_block::clst");

    //  Keep track of checked out blocks
    typedef std::map<size_t, dense_tensor_i<k_ordera, double>*> coba_map;
    typedef std::map<size_t, dense_tensor_i<k_orderb, double>*> cobb_map;
    coba_map coba;
    cobb_map cobb;

    //  Tensor contraction operation
    std::auto_ptr< tod_contract2<N, M, K> > op;

    //  Go through the contraction list and prepare the contraction
    for(typename contr_list::const_iterator i = clst.begin();
        i != clst.end(); ++i) {

        index<k_ordera> ia;
        index<k_orderb> ib;
        abs_index<k_ordera>::get_index(i->aia, m_bidimsa, ia);
        abs_index<k_orderb>::get_index(i->aib, m_bidimsb, ib);

        bool zeroa = ca.req_is_zero_block(ia);
        bool zerob = cb.req_is_zero_block(ib);
        if(zeroa || zerob) continue;

        if(coba.find(i->aia) == coba.end()) {
            dense_tensor_i<k_ordera, double> &ta = ca.req_block(ia);
            coba[i->aia] = &ta;
        }
        if(cobb.find(i->aib) == cobb.end()) {
            dense_tensor_i<k_orderb, double> &tb = cb.req_block(ib);
            cobb[i->aib] = &tb;
        }
        dense_tensor_i<k_ordera, double> &ta = *coba[i->aia];
        dense_tensor_i<k_orderb, double> &tb = *cobb[i->aib];

        tensor_transf<k_ordera, double> trainv(i->tra);
        trainv.invert();
        tensor_transf<k_orderb, double> trbinv(i->trb);
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
        if(zero) tod_set<k_orderc>().perform(tc);
    } else {
        op->perform(zero, c, tc);
    }

    //  Return input blocks
    for(typename coba_map::iterator i = coba.begin(); i != coba.end(); ++i) {
        index<k_ordera> ia;
        abs_index<k_ordera>::get_index(i->first, m_bidimsa, ia);
        ca.ret_block(ia);
    }
    for(typename cobb_map::iterator i = cobb.begin(); i != cobb.end(); ++i) {
        index<k_orderb> ib;
        abs_index<k_orderb>::get_index(i->first, m_bidimsb, ib);
        cb.ret_block(ib);
    }
}


template<size_t N, typename T>
btod_contract2_copy_task<N, T>::btod_contract2_copy_task(
    block_tensor_i<N, T> &bta, block_tensor_i<N, T> &btb, const index<N> &idx,
    bool zero) :

    m_bta(bta), m_btb(btb), m_idx(idx), m_zero(zero) {

}


template<size_t N, typename T>
void btod_contract2_copy_task<N, T>::perform() {

    block_tensor_ctrl<N, T> ca(m_bta);
    block_tensor_ctrl<N, T> cb(m_btb);

    bool zero = m_zero || cb.req_is_zero_block(m_idx);

    dense_tensor_i<N, T> &ta = ca.req_block(m_idx);
    dense_tensor_i<N, T> &tb = cb.req_block(m_idx);
    tod_copy<N>(ta).perform(zero, 1.0, tb);
    cb.ret_block(m_idx);
    ca.ret_block(m_idx);
}


template<size_t N, typename T>
btod_contract2_copy_task_iterator<N, T>::btod_contract2_copy_task_iterator(
    block_tensor_i<N, T> &bta, block_tensor_i<N, T> &btb, bool zero,
    const std::vector<size_t> &bi) :

    m_bta(bta), m_btb(btb), m_zero(zero), m_bi(bi), m_i(m_bi.begin()) {

}


template<size_t N, typename T>
bool btod_contract2_copy_task_iterator<N, T>::has_more() const {

    return m_i != m_bi.end();
}


template<size_t N, typename T>
libutil::task_i *btod_contract2_copy_task_iterator<N, T>::get_next() {

    dimensions<N> bidims = m_bta.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(*m_i, bidims, idx);
    btod_contract2_copy_task<N, T> *t =
        new btod_contract2_copy_task<N, T>(m_bta, m_btb, idx, m_zero);
    ++m_i;
    return t;
}


template<size_t N, typename T>
void btod_contract2_copy_task_observer<N, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
