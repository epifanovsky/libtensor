#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <map>
#include <memory>
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
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_set.h>
#include "bad_block_index_space.h"
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
    bto_contract2_clst<N, M, K, double> clstop(m_contr, m_bta, m_btb, m_bidimsa,
        m_bidimsb, m_bidimsc, idxc);
    clstop.build_list(false); // Build full contraction list
    const contr_list &clst = clstop.get_clst();

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

        contraction2<N, M, K> contr(m_contr);
        contr.permute_a(i->tra.get_perm());
        contr.permute_b(i->trb.get_perm());
        contr.permute_c(trc.get_perm());

        double kc = i->tra.get_scalar_tr().get_coeff() *
            i->trb.get_scalar_tr().get_coeff() *
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


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
