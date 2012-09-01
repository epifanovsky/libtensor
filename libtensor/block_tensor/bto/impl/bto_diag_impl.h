#ifndef LIBTENSOR_BTO_DIAG_IMPL_H
#define LIBTENSOR_BTO_DIAG_IMPL_H

#include <libtensor/core/allocator.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_permute.h>
#include "bto_aux_add_impl.h"
#include "bto_aux_copy_impl.h"
#include "../bto_diag.h"

namespace libtensor {


template<size_t N, size_t M, typename Traits>
const char *bto_diag<N, M, Traits>::k_clazz = "bto_diag<N, M, Traits>";


template<size_t N, size_t M, typename Traits>
class bto_diag_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N - M + 1>::type
        block_tensor_b_type;

private:
    bto_diag<N, M, Traits> &m_bto;
    block_tensor_b_type &m_btb;
    index<N - M + 1> m_idx;
    bto_stream_i<N - M + 1, Traits> &m_out;

public:
    bto_diag_task(
        bto_diag<N, M, Traits> &bto,
        block_tensor_b_type &btb,
        const index<N - M + 1> &idx,
        bto_stream_i<N - M + 1, Traits> &out);

    virtual ~bto_diag_task() { }
    virtual void perform();

};


template<size_t N, size_t M, typename Traits>
class bto_diag_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N - M + 1>::type
        block_tensor_b_type;

private:
    bto_diag<N, M, Traits> &m_bto;
    block_tensor_b_type &m_btb;
    bto_stream_i<N - M + 1, Traits> &m_out;
    const assignment_schedule<N - M + 1, element_type> &m_sch;
    typename assignment_schedule<N - M + 1, element_type>::iterator m_i;

public:
    bto_diag_task_iterator(
        bto_diag<N, M, Traits> &bto,
        block_tensor_b_type &btb,
        bto_stream_i<N - M + 1, Traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename Traits>
class bto_diag_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, typename Traits>
bto_diag<N, M, Traits>::bto_diag(block_tensora_t &bta, const mask<N> &m,
    const scalar_tr_t &c) :

    m_bta(bta), m_msk(m), m_tr(permutation<k_orderb>(), c),
    m_bis(mk_bis(bta.get_bis(), m_msk)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M, typename Traits>
bto_diag<N, M, Traits>::bto_diag(block_tensora_t &bta, const mask<N> &m,
    const permutation<N - M + 1> &p, const scalar_tr_t &c) :

    m_bta(bta), m_msk(m), m_tr(p, c),
    m_bis(mk_bis(bta.get_bis(), m_msk).permute(p)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims())  {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::sync_on() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::sync_off() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::perform(bto_stream_i<N - M + 1, Traits> &out) {

    typedef allocator<element_t> allocator_type;

    try {

        out.open();

        // TODO: replace with temporary block tensor from traits
        block_tensor<N - M + 1, element_t, allocator_type> btb(m_bis);
        block_tensor_ctrl<N - M + 1, element_t> cb(btb);
        cb.req_sync_on();
        sync_on();

        bto_diag_task_iterator<N, M, Traits> ti(*this, btb, out);
        bto_diag_task_observer<N, M, Traits> to;
        libutil::thread_pool::submit(ti, to);

        cb.req_sync_off();
        sync_off();

        out.close();

    } catch(...) {
        throw;
    }
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::perform(block_tensorb_t &btb) {

    bto_aux_copy<N - M + 1, Traits> out(m_sym, btb);
    perform(out);
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::perform(block_tensorb_t &btb, const element_t &c) {

    typedef typename Traits::template block_tensor_ctrl_type<N - M + 1>::type
        block_tensor_ctrl_b_type;

    block_tensor_ctrl_b_type cb(btb);
    addition_schedule<N - M + 1, Traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(m_sch, cb);

    bto_aux_add<N - M + 1, Traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::compute_block(bool zero, blockb_t &blk,
    const index<k_orderb> &ib, const tensorb_tr_t &trb, const element_t &c) {

    compute_block(blk, ib, trb, zero, c);
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::compute_block(blockb_t &blk,
    const index<k_orderb> &ib, const tensorb_tr_t &trb,
    bool zero, const element_t &c) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template to_diag_type<N, M>::type to_diag_t;

    bto_diag<N, M, Traits>::start_timer();

    try {

        block_tensor_ctrl_t ctrla(m_bta);
        dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

        //  Build ia from ib
        //
        sequence<k_ordera, size_t> map(0);
        size_t j = 0, jd; // Current index, index on diagonal
        bool b = false;
        for(size_t i = 0; i < k_ordera; i++) {
            if(m_msk[i]) {
                if(!b) { map[i] = jd = j++; b = true; }
                else { map[i] = jd; }
            } else {
                map[i] = j++;
            }
        }
        index<k_ordera> ia;
        index<k_orderb> ib2(ib);
        permutation<k_orderb> pinvb(m_tr.get_perm(), true);
        ib2.permute(pinvb);
        for(size_t i = 0; i < k_ordera; i++) ia[i] = ib2[map[i]];

        //  Find canonical index cia, transformation cia->ia
        //
        orbit<k_ordera, element_t> oa(ctrla.req_const_symmetry(), ia);
        abs_index<k_ordera> acia(oa.get_abs_canonical_index(), bidimsa);
        const tensora_tr_t &tra = oa.get_transf(ia);

        //  Build new diagonal mask and permutation in b
        //
        mask<k_ordera> m1(m_msk), m2(m_msk);
        sequence<k_ordera, size_t> map1(map), map2(map);
        m2.permute(tra.get_perm());
        tra.get_perm().apply(map2);

        sequence<N - M, size_t> seq1(0), seq2(0);
        sequence<k_orderb, size_t> seqb1(0), seqb2(0);
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
            if(!m1[i]) seq1[j1++] = map1[i];
            if(!m2[i]) seq2[j2++] = map2[i];
        }
        bool b1 = false, b2 = false;
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_orderb; i++) {
            if(m1[i] && !b1) { seqb1[i] = k_orderb; b1 = true; }
            else { seqb1[i] = seq1[j1++]; }
            if(m2[i] && !b2) { seqb2[i] = k_orderb; b2 = true; }
            else { seqb2[i] = seq2[j2++]; }
        }

        permutation_builder<k_orderb> pb(seqb2, seqb1);
        permutation<k_orderb> permb(pb.get_perm());
        permb.permute(m_tr.get_perm());
        permb.permute(permutation<k_orderb>(trb.get_perm(), true));

        //  Invoke the tensor operation
        //
        blocka_t &blka = ctrla.req_block(acia.get_index());

        scalar_tr_t sa(tra.get_scalar_tr());
        sa.invert().transform(m_tr.get_scalar_tr());
        sa.transform(trb.get_scalar_tr());

        if(zero) {
            sa.transform(scalar_tr_t(c));
            to_diag_t(blka, m2, permb, sa.get_coeff()).perform(blk);
        }
        else {
            to_diag_t(blka, m2, permb, sa.get_coeff()).perform(blk, c);
        }
        ctrla.ret_block(acia.get_index());

    }
    catch (...) {
        bto_diag<N, M, Traits>::stop_timer();
        throw;
    }

    bto_diag<N, M, Traits>::stop_timer();

}


template<size_t N, size_t M, typename Traits>
block_index_space<N - M + 1> bto_diag<N, M, Traits>::mk_bis(
    const block_index_space<N> &bis, const mask<N> &msk) {

    static const char *method =
        "mk_bis(const block_index_space<N>&, const mask<N>&)";

    //  Create the mask for the subspace builder
    //
    mask<N> m;
    bool b = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            if(!b) { m[i] = true; b = true; }
        } else {
            m[i] = true;
        }
    }

    //  Build the output block index space
    //
    block_index_subspace_builder<N - M + 1, M - 1> bb(bis, m);
    block_index_space<k_orderb> obis(bb.get_bis());
    obis.match_splits();

    return obis;
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::make_symmetry() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ca(m_bta);

    block_index_space<k_orderb> bis(m_bis);
    permutation<k_orderb> pinv(m_tr.get_perm(), true);
    bis.permute(pinv);
    symmetry<k_orderb, element_t> symx(bis);
    so_merge<N, M - 1, element_t>(ca.req_const_symmetry(),
            m_msk, sequence<N, size_t>()).perform(symx);
    so_permute<k_orderb, element_t>(symx, m_tr.get_perm()).perform(m_sym);

}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::make_schedule() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<k_orderb> pinv(m_tr.get_perm(), true);
    size_t map[k_ordera];
    size_t j = 0, jd;
    bool b = false;
    for(size_t i = 0; i < k_ordera; i++) {
        if(m_msk[i]) {
            if(b) map[i] = jd;
            else { map[i] = jd = j++; b = true; }
        } else {
            map[i] = j++;
        }
    }

    orbit_list<k_ordera, element_t> ola(ctrla.req_const_symmetry());
    orbit_list<k_orderb, element_t> olb(m_sym);
    for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<k_ordera> idxa;
        index<k_orderb> idxb(olb.get_index(iob));
        idxb.permute(pinv);

        for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
        if(! ola.contains(oa.get_abs_canonical_index())) continue;

        abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);

        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }
}


template<size_t N, size_t M, typename Traits>
bto_diag_task<N, M, Traits>::bto_diag_task(bto_diag<N, M, Traits> &bto,
    block_tensor_b_type &btb, const index<N - M + 1> &idx,
    bto_stream_i<N - M + 1, Traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, typename Traits>
void bto_diag_task<N, M, Traits>::perform() {

    typedef typename Traits::template block_tensor_ctrl_type<N - M + 1>::type
        block_tensor_ctrl_b_type;
    typedef typename Traits::template block_type<N - M + 1>::type block_b_type;
    typedef tensor_transf<N - M + 1, element_type> tensor_transf_b_type;

    block_tensor_ctrl_b_type cb(m_btb);
    block_b_type &blk = cb.req_block(m_idx);
    tensor_transf_b_type tr0;
    m_bto.compute_block(true, blk, m_idx, tr0, Traits::identity());
    m_out.put(m_idx, blk, tr0);
    cb.ret_block(m_idx);
    cb.req_zero_block(m_idx);
}


template<size_t N, size_t M, typename Traits>
bto_diag_task_iterator<N, M, Traits>::bto_diag_task_iterator(
    bto_diag<N, M, Traits> &bto, block_tensor_b_type &btb,
    bto_stream_i<N - M + 1, Traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, typename Traits>
bool bto_diag_task_iterator<N, M, Traits>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, typename Traits>
libutil::task_i *bto_diag_task_iterator<N, M, Traits>::get_next() {

    dimensions<N - M + 1> bidims = m_btb.get_bis().get_block_index_dims();
    index<N - M + 1> idx;
    abs_index<N - M + 1>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    bto_diag_task<N, M, Traits> *t =
        new bto_diag_task<N, M, Traits>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename Traits>
void bto_diag_task_observer<N, M, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DIAG_IMPL_H
