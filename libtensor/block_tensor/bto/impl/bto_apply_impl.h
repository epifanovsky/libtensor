#ifndef LIBTENSOR_BTO_APPLY_IMPL_H
#define LIBTENSOR_BTO_APPLY_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_apply.h>
#include "bto_aux_add_impl.h"
#include "bto_aux_copy_impl.h"
#include "../bto_apply.h"

namespace libtensor {


template<size_t N, typename Functor, typename Traits>
const char *bto_apply<N, Functor, Traits>::k_clazz =
    "bto_apply<N, Functor, Traits>";


template<size_t N, typename Functor, typename Traits>
class bto_apply_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    bto_apply<N, Functor, Traits> &m_bto;
    block_tensor_type &m_btb;
    index<N> m_idx;
    bto_stream_i<N, Traits> &m_out;

public:
    bto_apply_task(
        bto_apply<N, Functor, Traits> &bto,
        block_tensor_type &btb,
        const index<N> &idx,
        bto_stream_i<N, Traits> &out);

    virtual ~bto_apply_task() { }
    virtual void perform();

};


template<size_t N, typename Functor, typename Traits>
class bto_apply_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    bto_apply<N, Functor, Traits> &m_bto;
    block_tensor_type &m_btb;
    bto_stream_i<N, Traits> &m_out;
    const assignment_schedule<N, element_type> &m_sch;
    typename assignment_schedule<N, element_type>::iterator m_i;

public:
    bto_apply_task_iterator(
        bto_apply<N, Functor, Traits> &bto,
        block_tensor_type &btb,
        bto_stream_i<N, Traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Functor, typename Traits>
class bto_apply_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Functor, typename Traits>
bto_apply<N, Functor, Traits>::bto_apply(block_tensor_type &bta,
    const functor_type &fn, const scalar_transf_type &tr1,
    const tensor_transf_type &tr2) :

    m_bta(bta), m_fn(fn), m_tr1(tr1), m_tr2(tr2), m_bis(m_bta.get_bis()),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    //! Type of block tensor control object
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type ctrla(m_bta);
    so_apply<N, element_type>(ctrla.req_const_symmetry(), m_tr2.get_perm(),
            m_fn.transf(true), m_fn.transf(false),
            m_fn.keep_zero()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Functor, typename Traits>
bto_apply<N, Functor, Traits>::bto_apply(
        block_tensor_type &bta, const functor_type &fn,
        const permutation<N> &p, const scalar_transf_type &c) :

    m_bta(bta), m_fn(fn), m_tr1(c), m_tr2(p), m_bis(mk_bis(m_bta.get_bis(), p)),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    //! Type of block tensor control object
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type ctrla(m_bta);
    so_apply<N, element_type>(ctrla.req_const_symmetry(), m_tr2.get_perm(),
            m_fn.transf(true), m_fn.transf(false),
            m_fn.keep_zero()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::sync_on() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::sync_off() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::perform(bto_stream_i<N, Traits> &out) {

    typedef allocator<element_type> allocator_type;

    try {

        out.open();

        // TODO: replace with temporary block tensor from traits
        block_tensor<N, element_type, allocator_type> btb(m_bis);
        block_tensor_ctrl<N, element_type> cb(btb);
        cb.req_sync_on();
        sync_on();

        bto_apply_task_iterator<N, Functor, Traits> ti(*this, btb, out);
        bto_apply_task_observer<N, Functor, Traits> to;
        libutil::thread_pool::submit(ti, to);

        cb.req_sync_off();
        sync_off();

        out.close();

    } catch(...) {
        throw;
    }
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::perform(block_tensor_type &btb) {

    bto_aux_copy<N, Traits> out(m_sym, btb);
    perform(out);
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::perform(block_tensor_type &btb,
    const element_type &c) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type cb(btb);
    addition_schedule<N, Traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(m_sch, cb);

    bto_aux_add<N, Traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::compute_block(bool zero, block_type &blk,
    const index<N> &ib, const tensor_transf_type &tr, const element_type &c) {

    static const char *method =
            "compute_block(bool, block_type&, const index<N>&, "
            "const tensor_transf_type&, const element_type&)";

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template temp_block_type<N>::type tensor_type;
    typedef typename Traits::template to_set_type<N>::type to_set_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy_type;
    typedef typename Traits::template to_apply_type<N, Functor>::type
        to_apply_type;

    if(zero) to_set_type().perform(blk);

    block_tensor_ctrl_type ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr2.get_perm(), true);

    //  Corresponding index in A
    index<N> ia(ib);
    ia.permute(pinv);

    // Find the orbit the index belongs to
    orbit<N, element_type> oa(ctrla.req_const_symmetry(), ia);

    // If the orbit of A is not allowed, we assume it all elements are 0.0
    if (! oa.is_allowed()) {
        if (! m_fn.keep_zero()) {
            tensor_type tblk(blk.get_dims());
            to_set_type(m_fn(Traits::zero()) * c).perform(tblk);
            to_copy_type(tblk).perform(false, Traits::identity(), blk);
        }
        return;
    }

    //  Find the canonical index in A
    abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);

    //  Transformation for block from canonical A to B

    const tensor_transf_type &tra = oa.get_transf(ia);
    scalar_transf_type tr1(tra.get_scalar_tr());
    tensor_transf_type tr2(tra.get_perm());

    tr1.transform(m_tr1);
    tr2.transform(m_tr2);
    tr2.transform(scalar_transf_type(c)).
            transform(tensor_transf_type(tr, true));

    if(! ctrla.req_is_zero_block(acia.get_index())) {

        block_type &blka = ctrla.req_block(acia.get_index());
        to_apply_type(blka, m_fn, tr1, tr2).perform(false, blk);
        ctrla.ret_block(acia.get_index());
    }
    else {
        if (! m_fn.keep_zero()) {
            tensor_type tblk(blk.get_dims());
            element_type val = m_fn(Traits::zero());
            tr2.apply(val);
            to_set_type(val).perform(tblk);
            to_copy_type(tblk).perform(false, Traits::identity(), blk);
        }
    }
}


template<size_t N, typename Functor, typename Traits>
void bto_apply<N, Functor, Traits>::make_schedule() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr2.get_perm(), ! m_tr2.get_perm().is_identity());

    orbit_list<N, element_type> ol(m_sym);
    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
            io != ol.end(); io++) {

        // If m_fn(0.0) yields 0.0 only non-zero blocks of tensor A need to
        // be considered
        if (m_fn.keep_zero()) {
            index<N> ia(ol.get_index(io)); ia.permute(pinv);

            orbit<N, element_type> oa(ctrla.req_const_symmetry(), ia);
            if (! oa.is_allowed()) continue;

            abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);
            if (ctrla.req_is_zero_block(acia.get_index())) continue;

            m_sch.insert(ol.get_abs_index(io));
        }
        else {
            m_sch.insert(ol.get_abs_index(io));
        }
    }
}


template<size_t N, typename Functor, typename Traits>
block_index_space<N> bto_apply<N, Functor, Traits>::mk_bis(
    const block_index_space<N> &bis, const permutation<N> &perm) {

    block_index_space<N> bis1(bis);
    bis1.permute(perm);
    return bis1;
}


template<size_t N, typename Functor, typename Traits>
bto_apply_task<N, Functor, Traits>::bto_apply_task(
    bto_apply<N, Functor, Traits> &bto, block_tensor_type &btb,
    const index<N> &idx, bto_stream_i<N, Traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, typename Functor, typename Traits>
void bto_apply_task<N, Functor, Traits>::perform() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

    block_tensor_ctrl_type cb(m_btb);
    block_type &blk = cb.req_block(m_idx);
    tensor_transf_type tr0;
    m_bto.compute_block(true, blk, m_idx, tr0, Traits::identity());
    m_out.put(m_idx, blk, tr0);
    cb.ret_block(m_idx);
    cb.req_zero_block(m_idx);
}


template<size_t N, typename Functor, typename Traits>
bto_apply_task_iterator<N, Functor, Traits>::bto_apply_task_iterator(
    bto_apply<N, Functor, Traits> &bto, block_tensor_type &btb,
    bto_stream_i<N, Traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename Functor, typename Traits>
bool bto_apply_task_iterator<N, Functor, Traits>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename Functor, typename Traits>
libutil::task_i *bto_apply_task_iterator<N, Functor, Traits>::get_next() {

    dimensions<N> bidims = m_btb.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    bto_apply_task<N, Functor, Traits> *t =
        new bto_apply_task<N, Functor, Traits>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Functor, typename Traits>
void bto_apply_task_observer<N, Functor, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_IMPL_H
