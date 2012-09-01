#ifndef LIBTENSOR_BTOD_EXTRACT_IMPL_H
#define LIBTENSOR_BTOD_EXTRACT_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_extract.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/symmetry/so_permute.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include "../btod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char *btod_extract<N, M>::k_clazz = "btod_extract<N, M>";


template<size_t N, size_t M, typename T>
class btod_extract_task : public libutil::task_i {
private:
    btod_extract<N, M> &m_bto;
    block_tensor_i<N - M, T> &m_btc;
    index<N - M> m_idx;
    bto_stream_i<N - M, btod_traits> &m_out;

public:
    btod_extract_task(
        btod_extract<N, M> &bto,
        block_tensor_i<N - M, T> &btc,
        const index<N - M> &idx,
        bto_stream_i<N - M, btod_traits> &out);

    virtual ~btod_extract_task() { }
    virtual void perform();

};


template<size_t N, size_t M, typename T>
class btod_extract_task_iterator : public libutil::task_iterator_i {
private:
    btod_extract<N, M> &m_bto;
    block_tensor_i<N - M, T> &m_btc;
    bto_stream_i<N - M, btod_traits> &m_out;
    const assignment_schedule<N - M, double> &m_sch;
    typename assignment_schedule<N - M, double>::iterator m_i;

public:
    btod_extract_task_iterator(
        btod_extract<N, M> &bto,
        block_tensor_i<N - M, T> &btc,
        bto_stream_i<N - M, btod_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename T>
class btod_extract_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
    const mask<N> &m, const index<N> &idxbl, const index<N> &idxibl,
    double c) :

    m_bta(bta), m_msk(m), m_idxbl(idxbl), m_idxibl(idxibl), m_c(c),
    m_bis(mk_bis(bta.get_bis(), m_msk, permutation<N - M>())), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    block_tensor_ctrl<N, double> ctrla(bta);

    sequence<N, size_t> seq(0);
    mask<N> invmsk;
    for (register size_t i = 0, j = 0; i < N; i++) {
        invmsk[i] = !m_msk[i];
        if (invmsk[i]) seq[i] = j++;
    }

    so_reduce<N, M, double>(ctrla.req_const_symmetry(),
            invmsk, seq, index_range<N>(idxbl, idxbl),
            index_range<N>(idxibl, idxibl)).perform(m_sym);

    make_schedule();
}


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
    const mask<N> &m, const permutation<N - M> &perm,
    const index<N> &idxbl, const index<N> &idxibl, double c) :

    m_bta(bta), m_msk(m), m_perm(perm), m_idxbl(idxbl), m_idxibl(idxibl),
    m_c(c), m_bis(mk_bis(bta.get_bis(), m_msk, perm)), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    permutation<N - M> pinv(perm, true);
    block_index_space<N - M> bisinv(m_bis);
    bisinv.permute(pinv);

    block_tensor_ctrl<N, double> ctrla(bta);
    symmetry<k_orderb, double> sym(bisinv);

    sequence<N, size_t> seq(0);
    mask<N> invmsk;
    for (register size_t i = 0, j = 0; i < N; i++) {
        invmsk[i] = !m_msk[i];
        if (invmsk[i]) seq[i] = j++;
    }

    so_reduce<N, M, double>(ctrla.req_const_symmetry(),
            invmsk, seq, index_range<N>(idxbl, idxbl),
            index_range<N>(idxibl, idxibl)).perform(sym);
    so_permute<k_orderb, double>(sym, perm).perform(m_sym);

    make_schedule();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_on() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_off() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(bto_stream_i<N - M, btod_traits> &out) {

    typedef allocator<double> allocator_type;

    try {

        out.open();

        block_tensor<N - M, double, allocator_type> btc(m_bis);
        block_tensor_ctrl<N - M, double> cc(btc);
        cc.req_sync_on();
        sync_on();

        btod_extract_task_iterator<N, M, double> ti(*this, btc, out);
        btod_extract_task_observer<N, M, double> to;
        libutil::thread_pool::submit(ti, to);

        cc.req_sync_off();
        sync_off();

        out.close();

    } catch(...) {
        throw;
    }
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(block_tensor_i<N - M, double> &btb) {

    bto_aux_copy<N - M, btod_traits> out(m_sym, btb);
    perform(out);
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(block_tensor_i<N - M, double> &btb,
    const double &c) {

    typedef block_tensor_ctrl<N - M, double> block_tensor_ctrl_type;

    block_tensor_ctrl_type cb(btb);
    addition_schedule<N - M, btod_traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N - M, btod_traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(bool zero,
    dense_tensor_i<k_orderb, double> &blk, const index<k_orderb> &idx,
    const tensor_transf<k_orderb, double> &tr, const double &c) {

    do_compute_block(blk, idx, tr, c, zero);
}


template<size_t N, size_t M>
void btod_extract<N, M>::do_compute_block(dense_tensor_i<k_orderb, double> &blk,
    const index<k_orderb> &idx, const tensor_transf<k_orderb, double> &tr,
    double c, bool zero) {

    btod_extract<N, M>::start_timer();

    block_tensor_ctrl<k_ordera, double> ctrla(m_bta);

    permutation<k_orderb> pinv(m_perm, true);

    index<k_ordera> idxa;
    index<k_orderb> idxb(idx);

    idxb.permute(pinv);

    for(size_t i = 0, j = 0; i < k_ordera; i++) {
        if(m_msk[i]) {
            idxa[i] = idxb[j++];
        } else {
            idxa[i] = m_idxbl[i];
        }
    }

    orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

    abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    tensor_transf<k_ordera, double> tra(oa.get_transf(idxa)); tra.invert();

    mask<k_ordera> msk1(m_msk), msk2(m_msk);
    msk2.permute(tra.get_perm());

    sequence<k_ordera, size_t> seqa1(0), seqa2(0);
    sequence<k_orderb, size_t> seqb1(0), seqb2(0);
    for(register size_t i = 0; i < k_ordera; i++) seqa2[i] = seqa1[i] = i;
    tra.get_perm().apply(seqa2);
    for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
        if(msk1[i]) seqb1[j1++] = seqa1[i];
        if(msk2[i]) seqb2[j2++] = seqa2[i];
    }

    permutation_builder<k_orderb> pb(seqb2, seqb1);
    permutation<k_orderb> permb(pb.get_perm());
    permb.permute(m_perm);
    permb.permute(tr.get_perm());

    index<k_ordera> idxibl2(m_idxibl);
    idxibl2.permute(tra.get_perm());

    bool zeroa = !oa.is_allowed();
    if(!zeroa) zeroa = ctrla.req_is_zero_block(cidxa.get_index());

    if(!zeroa) {

        dense_tensor_i<k_ordera, double> &blka = ctrla.req_block(
            cidxa.get_index());
        if(zero) {
            tod_extract<N, M>(blka, msk2, permb, idxibl2,
                tra.get_scalar_tr().get_coeff() * m_c * c).perform(blk);
        } else {
            tod_extract<N, M>(blka, msk2, permb, idxibl2,
                tra.get_scalar_tr().get_coeff() * m_c).perform(blk, c);
        }
        ctrla.ret_block(cidxa.get_index());
    } else {

        if(zero) tod_set<N - M>().perform(blk);
    }

    btod_extract<N, M>::stop_timer();
}


template<size_t N, size_t M>
block_index_space<N - M> btod_extract<N, M>::mk_bis(
    const block_index_space<N> &bis, const mask<N> &msk,
    const permutation<N - M> &perm) {

    static const char *method = "mk_bis(const block_index_space<N>&, "
        "const mask<N>&, const permutation<N - M>&)";

    dimensions<N> idims(bis.get_dims());

    //  Compute output dimensions
    //

    index<k_orderb> i1, i2;

    size_t m = 0, j = 0;
    size_t map[k_orderb];//map between B and A

    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            i2[j] = idims[i] - 1;
            map[j] = i;
            j++;
        } else {
            m++;
        }
    }


    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "m");
    }

    block_index_space<k_orderb> obis(dimensions<k_orderb>(
        index_range<k_orderb>(i1, i2)));

    mask<k_orderb> msk_done;
    bool done = false;
    while(!done) {

        size_t i = 0;
        while(i < k_orderb && msk_done[i]) i++;
        if(i == k_orderb) {
            done = true;
            continue;
        }
        size_t typ = bis.get_type(map[i]);
        const split_points &splits = bis.get_splits(typ);
        mask<k_orderb> msk_typ;
        for(size_t k = 0; k < k_orderb; k++) {
            if(bis.get_type(map[k]) == typ) msk_typ[k] = true;
        }
        size_t npts = splits.get_num_points();
        for(register size_t k = 0; k < npts; k++) {
            obis.split(msk_typ, splits[k]);
        }
        msk_done |= msk_typ;
    }

    obis.permute(perm);
    return obis;
}


template<size_t N, size_t M>
void btod_extract<N, M>::make_schedule() {

    btod_extract<N, M>::start_timer("make_schedule");

    block_tensor_ctrl<N, double> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<k_orderb> pinv(m_perm, true);

    orbit_list<k_orderb, double> olb(m_sym);
    for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<k_ordera> idxa;
        index<k_orderb> idxb(olb.get_index(iob));

        idxb.permute(pinv);

        for(size_t i = 0, j = 0; i < k_ordera; i++) {
            if(m_msk[i]) idxa[i] = idxb[j++];
            else idxa[i] = m_idxbl[i];
        }

        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

        abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
                m_bta.get_bis().get_block_index_dims());

        if(!oa.is_allowed()) continue;
        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }

    btod_extract<N, M>::stop_timer("make_schedule");

}


template<size_t N, size_t M, typename T>
btod_extract_task<N, M, T>::btod_extract_task(btod_extract<N, M> &bto,
    block_tensor_i<N - M, T> &btc, const index<N - M> &idx,
    bto_stream_i<N - M, btod_traits> &out) :

    m_bto(bto), m_btc(btc), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, typename T>
void btod_extract_task<N, M, T>::perform() {

    block_tensor_ctrl<N - M, T> cc(m_btc);
    dense_tensor_i<N - M, T> &blk = cc.req_block(m_idx);
    tensor_transf<N - M, T> tr0;
    m_bto.compute_block(true, blk, m_idx, tr0, 1.0);
    m_out.put(m_idx, blk, tr0);
    cc.ret_block(m_idx);
    cc.req_zero_block(m_idx);
}


template<size_t N, size_t M, typename T>
btod_extract_task_iterator<N, M, T>::btod_extract_task_iterator(
    btod_extract<N, M> &bto, block_tensor_i<N - M, T> &btc,
    bto_stream_i<N - M, btod_traits> &out) :

    m_bto(bto), m_btc(btc), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, typename T>
bool btod_extract_task_iterator<N, M, T>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, typename T>
libutil::task_i *btod_extract_task_iterator<N, M, T>::get_next() {

    dimensions<N - M> bidims = m_btc.get_bis().get_block_index_dims();
    index<N - M> idx;
    abs_index<N - M>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    btod_extract_task<N, M, T> *t =
        new btod_extract_task<N, M, T>(m_bto, m_btc, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename T>
void btod_extract_task_observer<N, M, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_IMPL_H
