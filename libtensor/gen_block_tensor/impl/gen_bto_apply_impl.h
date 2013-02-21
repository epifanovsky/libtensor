#ifndef LIBTENSOR_BTO_APPLY_IMPL_H
#define LIBTENSOR_BTO_APPLY_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_apply.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_block_stream_i.h"
#include "../gen_bto_apply.h"

namespace libtensor {


template<size_t N, typename Functor, typename Traits, typename Timed>
const char *gen_bto_apply<N, Functor, Traits, Timed>::k_clazz =
    "gen_bto_apply<N, Functor, Traits, Timed>";


template<size_t N, typename Functor, typename Traits, typename Timed>
class gen_bto_apply_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_apply<N, Functor, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    index<N> m_idx;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_apply_task(
        gen_bto_apply<N, Functor, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        const index<N> &idx,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_apply_task() { }
    virtual void perform();

};


template<size_t N, typename Functor, typename Traits, typename Timed>
class gen_bto_apply_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_apply<N, Functor, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    gen_block_stream_i<N, bti_traits> &m_out;
    const assignment_schedule<N, element_type> &m_sch;
    typename assignment_schedule<N, element_type>::iterator m_i;

public:
    gen_bto_apply_task_iterator(
        gen_bto_apply<N, Functor, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Functor, typename Traits>
class gen_bto_apply_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Functor, typename Traits, typename Timed>
gen_bto_apply<N, Functor, Traits, Timed>::gen_bto_apply(
        gen_block_tensor_rd_i<N, bti_traits> &bta, const functor_type &fn,
        const scalar_transf_type &tr1, const tensor_transf_type &tr2) :

    m_bta(bta), m_fn(fn), m_tr1(tr1), m_tr2(tr2),
    m_bis(mk_bis(m_bta.get_bis(), tr2.get_perm())),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta);
    so_apply<N, element_type>(
            ctrla.req_const_symmetry(), m_tr2.get_perm(),
            m_fn.transf(true), m_fn.transf(false),
            m_fn.keep_zero()).perform(m_sym);

    make_schedule();
}


template<size_t N, typename Functor, typename Traits, typename Timed>
void gen_bto_apply<N, Functor, Traits, Timed>::perform(
        gen_block_stream_i<N, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

    gen_bto_apply::start_timer();

    try {

        // TODO: replace with temporary block tensor from traits
        temp_block_tensor_type btb(m_bis);

        gen_bto_apply_task_iterator<N, Functor, Traits, Timed> ti(*this,
                btb, out);
        gen_bto_apply_task_observer<N, Functor, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_apply::stop_timer();
        throw;
    }

    gen_bto_apply::stop_timer();
}


template<size_t N, typename Functor, typename Traits, typename Timed>
void gen_bto_apply<N, Functor, Traits, Timed>::compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf_type &trb,
        wr_block_type &blkb) {

    gen_bto_apply::start_timer("compute_block");
    try {

        compute_block_untimed(zero, ib, trb, blkb);

    } catch (...) {
        gen_bto_apply::stop_timer("compute_block");
        throw;
    }

    gen_bto_apply::stop_timer("compute_block");
}


template<size_t N, typename Functor, typename Traits, typename Timed>
void gen_bto_apply<N, Functor, Traits, Timed>::compute_block_untimed(
        bool zero,
        const index<N> &ib,
        const tensor_transf_type &trb,
        wr_block_type &blkb) {

    typedef typename Traits::template temp_block_type<N>::type temp_block_type;
    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_copy_type<N>::type to_copy;
    typedef typename Traits::template to_apply_type<N, Functor>::type
        to_apply;

    //if(zero) to_set_type().perform(blk);

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr2.get_perm(), true);

    //  Corresponding index in A
    index<N> ia(ib);
    ia.permute(pinv);

    // Find the orbit the index belongs to
    orbit<N, element_type> oa(ctrla.req_const_symmetry(), ia);

    // If the orbit of A is not allowed, we assume all its elements are 0.0
    if (! oa.is_allowed()) {
        if (! m_fn.keep_zero()) {
            element_type val = m_fn(Traits::zero());
            m_tr2.apply(val);

            if (zero)
                to_set(val).perform(blkb);
            else {
                temp_block_type temp_blk(blkb.get_dims());
                to_set(val).perform(temp_blk);
                to_copy(temp_blk).perform(false, blkb);
            }
        }
        else {
            to_set().perform(blkb);
        }
        return;
    }

    //  Find the canonical index in A
    abs_index<N> acia(oa.get_acindex(), bidimsa);

    //  Transformation for block from canonical A to B
    const tensor_transf_type &tra = oa.get_transf(ia);

    scalar_transf_type tr1(tra.get_scalar_tr());
    tensor_transf_type tr2(tra.get_perm());
    tr1.transform(m_tr1);
    tr2.transform(m_tr2);
    tr2.transform(tensor_transf_type(trb, true));

    if(! ctrla.req_is_zero_block(acia.get_index())) {

        rd_block_type &blka = ctrla.req_const_block(acia.get_index());
        to_apply(blka, m_fn, tr1, tr2).perform(zero, blkb);
        ctrla.ret_const_block(acia.get_index());
    }
    else {

        if (! m_fn.keep_zero()) {
            element_type val = m_fn(Traits::zero());
            tr2.apply(val);

            if (zero)
                to_set(val).perform(blkb);
            else {
                temp_block_type temp_blk(blkb.get_dims());
                to_set(val).perform(temp_blk);
                to_copy(temp_blk).perform(false, blkb);
            }
        }
    }
}


template<size_t N, typename Functor, typename Traits, typename Timed>
block_index_space<N> gen_bto_apply<N, Functor, Traits, Timed>::mk_bis(
    const block_index_space<N> &bis, const permutation<N> &perm) {

    block_index_space<N> bis1(bis);
    bis1.permute(perm);
    return bis1;
}


template<size_t N, typename Functor, typename Traits, typename Timed>
void gen_bto_apply<N, Functor, Traits, Timed>::make_schedule() {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr2.get_perm(), true);

    orbit_list<N, element_type> ol(m_sym);
    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
            io != ol.end(); io++) {

        // If m_fn(0.0) yields 0.0 only non-zero blocks of tensor A need to
        // be considered
        if (m_fn.keep_zero()) {
            index<N> ia;
            ol.get_index(io, ia);
            ia.permute(pinv);

            orbit<N, element_type> oa(ctrla.req_const_symmetry(), ia);
            if (! oa.is_allowed()) continue;

            abs_index<N> acia(oa.get_acindex(), bidimsa);
            if (ctrla.req_is_zero_block(acia.get_index())) continue;

            m_sch.insert(ol.get_abs_index(io));
        }
        else {
            m_sch.insert(ol.get_abs_index(io));
        }
    }
}



template<size_t N, typename Functor, typename Traits, typename Timed>
gen_bto_apply_task<N, Functor, Traits, Timed>::gen_bto_apply_task(
    gen_bto_apply<N, Functor, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    const index<N> &idx,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, typename Functor, typename Traits, typename Timed>
void gen_bto_apply_task<N, Functor, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    tensor_transf<N, element_type> tr0;
    gen_block_tensor_ctrl<N, bti_traits> cb(m_btb);
    {
        wr_block_type &blkb = cb.req_block(m_idx);
        m_bto.compute_block_untimed(true, m_idx, tr0, blkb);
        cb.ret_block(m_idx);
    }

    {
        rd_block_type &blkb = cb.req_const_block(m_idx);
        m_out.put(m_idx, blkb, tr0);
        cb.ret_const_block(m_idx);
    }

    cb.req_zero_block(m_idx);
}


template<size_t N, typename Functor, typename Traits, typename Timed>
gen_bto_apply_task_iterator<N, Functor, Traits, Timed>::
gen_bto_apply_task_iterator(
    gen_bto_apply<N, Functor, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename Functor, typename Traits, typename Timed>
bool gen_bto_apply_task_iterator<N, Functor, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename Functor, typename Traits, typename Timed>
libutil::task_i *
gen_bto_apply_task_iterator<N, Functor, Traits, Timed>::get_next() {

    dimensions<N> bidims = m_btb.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_apply_task<N, Functor, Traits, Timed> *t =
        new gen_bto_apply_task<N, Functor, Traits, Timed>(m_bto,
                m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Functor, typename Traits>
void gen_bto_apply_task_observer<N, Functor, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_IMPL_H
