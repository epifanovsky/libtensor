#ifndef LIBTENSOR_BTO_COPY_IMPL_H
#define LIBTENSOR_BTO_COPY_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_permute.h>
#include "bto_aux_add_impl.h"
#include "bto_aux_copy_impl.h"
#include "../bto_copy.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *bto_copy<N, Traits>::k_clazz = "bto_copy<N, Traits>";


template<size_t N, typename Traits>
class bto_copy_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef bto_stream_i<N, Traits> bto_stream_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;
    typedef symmetry<N, element_type> symmetry_type;

private:
    block_tensor_type &m_bta;
    const tensor_transf_type &m_tra;
    const dimensions<N> &m_bidimsb;
    const symmetry_type &m_symb;
    bto_stream_type &m_out;
    index<N> m_ia;

public:
    bto_copy_task(
        block_tensor_type &bta,
        const tensor_transf_type &tra,
        const dimensions<N> &bidimsb,
        const symmetry_type &symb,
        const index<N> &ia,
        bto_stream_type &out);

    virtual ~bto_copy_task() { }
    virtual void perform();

};


template<size_t N, typename Traits>
class bto_copy_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef bto_stream_i<N, Traits> bto_stream_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;
    typedef symmetry<N, element_type> symmetry_type;

private:
    block_tensor_type &m_bta;
    tensor_transf_type m_tra;
    const symmetry_type &m_symb;
    bto_stream_type &m_out;
    dimensions<N> m_bidimsb;
    block_tensor_ctrl_type m_ca;
    orbit_list<N, element_type> m_ola;
    typename orbit_list<N, element_type>::iterator m_ioa;

public:
    bto_copy_task_iterator(
        block_tensor_type &bta,
        const tensor_transf_type &tra,
        const symmetry_type &symb,
        bto_stream_type &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

private:
    void skip_zero_blocks();

};


template<size_t N, typename Traits>
class bto_copy_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Traits>
bto_copy<N, Traits>::bto_copy(block_tensor_type &bta, const tensor_transf_t &tr) :

    m_bta(bta), m_tr(tr), m_bis(mk_bis(m_bta.get_bis(), tr.get_perm())),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    so_copy<N, element_type>(ctrla.req_const_symmetry()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Traits>
bto_copy<N, Traits>::bto_copy(block_tensor_type &bta, const scalar_transf_t &c) :

    m_bta(bta), m_tr(permutation<N>(), c), m_bis(m_bta.get_bis()),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    so_copy<N, element_type>(ctrla.req_const_symmetry()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Traits>
bto_copy<N, Traits>::bto_copy(block_tensor_type &bta, const permutation<N> &p,
    const scalar_transf_t &c) :

    m_bta(bta), m_tr(p, c), m_bis(mk_bis(m_bta.get_bis(), p)),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    so_permute<N, element_type>(ctrla.req_const_symmetry(),
            m_tr.get_perm()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::sync_on() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::sync_off() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::perform(block_tensor_type &btb) {

    bto_aux_copy<N, Traits> out(m_sym, btb);
    perform(out);
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::perform(block_tensor_type &btb,
    const element_type &c) {

    block_tensor_ctrl_type cb(btb);
    addition_schedule<N, Traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(m_sch, cb);

    bto_aux_add<N, Traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::perform(bto_stream_type &out) {

    bto_copy<N, Traits>::start_timer();

    try {

        out.open();

        bto_copy_task_iterator<N, Traits> ti(m_bta, m_tr, m_sym, out);
        bto_copy_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        bto_copy<N, Traits>::stop_timer();
        throw;
    }

    bto_copy<N, Traits>::stop_timer();
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::compute_block(bool zero, block_type &blk,
    const index<N> &ib, const tensor_transf_t &tr, const element_type &c) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template to_set_type<N>::type to_set_t;
    typedef typename Traits::template to_copy_type<N>::type to_copy_t;

    bto_copy<N, Traits>::start_timer("compute_block");

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    tensor_transf_t trinv(m_tr, true);

    //  Corresponding index in A
    index<N> ia(ib);
    ia.permute(trinv.get_perm());

    //  Find the canonical index in A
    orbit<N, double> oa(ctrla.req_const_symmetry(), ia);
    abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);


    //  Transformation for block from canonical A to B
    tensor_transf_t tra(oa.get_transf(ia));
    tra.transform(m_tr).transform(scalar_transf_t(c));
    tra.transform(tr);

    if(zero) to_set_t().perform(blk);
    if(!ctrla.req_is_zero_block(acia.get_index())) {
        block_type &blka = ctrla.req_block(acia.get_index());
        to_copy_t(blka, tra).perform(false, Traits::identity(), blk);
        ctrla.ret_block(acia.get_index());
    }

    bto_copy<N, Traits>::stop_timer("compute_block");
}


template<size_t N, typename Traits>
void bto_copy<N, Traits>::make_schedule() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    bto_copy<N, Traits>::start_timer("make_schedule");

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    bool identity = m_tr.get_perm().is_identity();

    orbit_list<N, element_type> ola(ctrla.req_const_symmetry());
    for(typename orbit_list<N, element_type>::iterator ioa = ola.begin();
        ioa != ola.end(); ioa++) {

        if(ctrla.req_is_zero_block(ola.get_index(ioa))) continue;

        if(!identity) {
            index<N> bib(ola.get_index(ioa)); bib.permute(m_tr.get_perm());
            orbit<N, element_type> ob(m_sym, bib);
            m_sch.insert(ob.get_abs_canonical_index());
        } else {
            m_sch.insert(ola.get_abs_index(ioa));
        }
    }

    bto_copy<N, Traits>::stop_timer("make_schedule");
}


template<size_t N, typename Traits>
block_index_space<N> bto_copy<N, Traits>::mk_bis(
        const block_index_space<N> &bis, const permutation<N> &perm) {

    block_index_space<N> bis1(bis);
    bis1.permute(perm);
    return bis1;
}


template<size_t N, typename Traits>
bto_copy_task<N, Traits>::bto_copy_task(block_tensor_type &bta,
    const tensor_transf_type &tra, const dimensions<N> &bidimsb,
    const symmetry_type &symb, const index<N> &ia, bto_stream_type &out) :

    m_bta(bta), m_tra(tra), m_bidimsb(bidimsb), m_symb(symb), m_ia(ia),
    m_out(out) {

}


template<size_t N, typename Traits>
void bto_copy_task<N, Traits>::perform() {

    block_tensor_ctrl_type ca(m_bta);

    block_type &ba = ca.req_block(m_ia);
    if(!m_tra.get_perm().is_identity()) {
        index<N> ib(m_ia);
        ib.permute(m_tra.get_perm());
        orbit<N, element_type> ob(m_symb, ib);
        abs_index<N> acib(ob.get_abs_canonical_index(), m_bidimsb);
        tensor_transf<N, element_type> trb(ob.get_transf(ib));
        trb.invert();
        tensor_transf<N, element_type> tra(m_tra);
        tra.transform(trb);
        m_out.put(acib.get_index(), ba, tra);
    } else {
        m_out.put(m_ia, ba, m_tra);
    }
    ca.ret_block(m_ia);
}


template<size_t N, typename Traits>
bto_copy_task_iterator<N, Traits>::bto_copy_task_iterator(
    block_tensor_type &bta, const tensor_transf_type &tra,
    const symmetry_type &symb, bto_stream_type &out) :

    m_bta(bta), m_tra(tra), m_symb(symb), m_out(out),
    m_bidimsb(m_symb.get_bis().get_block_index_dims()),
    m_ca(m_bta), m_ola(m_ca.req_const_symmetry()), m_ioa(m_ola.begin()) {

    skip_zero_blocks();
}


template<size_t N, typename Traits>
bool bto_copy_task_iterator<N, Traits>::has_more() const {

    return m_ioa != m_ola.end();
}


template<size_t N, typename Traits>
libutil::task_i *bto_copy_task_iterator<N, Traits>::get_next() {

    bto_copy_task<N, Traits> *t = new bto_copy_task<N, Traits>(m_bta, m_tra,
        m_bidimsb, m_symb, m_ola.get_index(m_ioa), m_out);
    ++m_ioa;
    skip_zero_blocks();
    return t;
}


template<size_t N, typename Traits>
void bto_copy_task_iterator<N, Traits>::skip_zero_blocks() {

    while(m_ioa != m_ola.end()) {
        if(!m_ca.req_is_zero_block(m_ola.get_index(m_ioa))) break;
        ++m_ioa;
    }
}


template<size_t N, typename Traits>
void bto_copy_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_IMPL_H
