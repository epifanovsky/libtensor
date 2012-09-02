#ifndef LIBTENSOR_BTOD_MULT_IMPL_H
#define LIBTENSOR_BTOD_MULT_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/dense_tensor/tod_mult.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../../bto/impl/bto_aux_add_impl.h"
#include "../../bto/impl/bto_aux_copy_impl.h"
#include "../btod_mult.h"

namespace libtensor {


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";


template<size_t N, typename T>
class btod_mult_task : public libutil::task_i {
private:
    btod_mult<N> &m_bto;
    block_tensor_i<N, T> &m_btc;
    index<N> m_idx;
    bto_stream_i<N, btod_traits> &m_out;

public:
    btod_mult_task(
        btod_mult<N> &bto,
        block_tensor_i<N, T> &btc,
        const index<N> &idx,
        bto_stream_i<N, btod_traits> &out);

    virtual ~btod_mult_task() { }
    virtual void perform();

};


template<size_t N, typename T>
class btod_mult_task_iterator : public libutil::task_iterator_i {
private:
    btod_mult<N> &m_bto;
    block_tensor_i<N, T> &m_btc;
    bto_stream_i<N, btod_traits> &m_out;
    const assignment_schedule<N, double> &m_sch;
    typename assignment_schedule<N, double>::iterator m_i;

public:
    btod_mult_task_iterator(
        btod_mult<N> &bto,
        block_tensor_i<N, T> &btc,
        bto_stream_i<N, btod_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename T>
class btod_mult_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N>
btod_mult<N>::btod_mult(block_tensor_i<N, double> &bta,
    block_tensor_i<N, double> &btb, bool recip, double c) :

    m_bta(bta), m_btb(btb), m_recip(recip), m_c(c), m_bis(m_bta.get_bis()),
    m_sym(m_bta.get_bis()), m_sch(m_bta.get_bis().get_block_index_dims()) {

    static const char *method = "btod_mult(block_tensor_i<N, double>&, "
        "block_tensor_i<N, double>&, bool)";

    if(! m_bta.get_bis().equals(m_btb.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta,btb");
    }

    block_tensor_ctrl<N, double> cbta(bta), cbtb(btb);
    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = i; seq2b[i] = m_pa[i];
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_pb[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(m_bis, m_bis,
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(cbta.req_const_symmetry(),
            cbtb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);

    make_schedule();
}

template<size_t N>
btod_mult<N>::btod_mult(
    block_tensor_i<N, double> &bta, const permutation<N> &pa,
    block_tensor_i<N, double> &btb, const permutation<N> &pb,
    bool recip, double c) :

    m_bta(bta), m_btb(btb), m_pa(pa), m_pb(pb), m_recip(recip), m_c(c),
    m_bis(block_index_space<N>(m_bta.get_bis()).permute(m_pa)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    static const char *method = "btod_mult(block_tensor_i<N, double>&, "
        "block_tensor_i<N, double>&, bool)";

    block_index_space<N> bisb(m_btb.get_bis());
    bisb.permute(m_pb);
    if(! m_bis.equals(bisb)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta,btb");
    }

    block_tensor_ctrl<N, double> cbta(bta), cbtb(btb);

    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = i; seq2b[i] = m_pa[i];
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_pb[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(m_bis, m_bis,
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(cbta.req_const_symmetry(),
            cbtb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);

    make_schedule();
}

template<size_t N>
btod_mult<N>::~btod_mult() {

}


template<size_t N>
void btod_mult<N>::sync_on() {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);
    ctrla.req_sync_on();
    ctrlb.req_sync_on();
}


template<size_t N>
void btod_mult<N>::sync_off() {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);
    ctrla.req_sync_off();
    ctrlb.req_sync_off();
}


template<size_t N>
void btod_mult<N>::perform(bto_stream_i<N, btod_traits> &out) {

    typedef allocator<double> allocator_type;

    try {

        out.open();

        block_tensor<N, double, allocator_type> btc(m_bis);
        block_tensor_ctrl<N, double> cc(btc);
        cc.req_sync_on();
        sync_on();

        btod_mult_task_iterator<N, double> ti(*this, btc, out);
        btod_mult_task_observer<N, double> to;
        libutil::thread_pool::submit(ti, to);

        cc.req_sync_off();
        sync_off();

        out.close();

    } catch(...) {
        throw;
    }
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc) {

    typedef btod_traits Traits;

    bto_aux_copy<N, Traits> out(m_sym, btc);
    perform(out);
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc, const double &d) {

    typedef btod_traits Traits;

    block_tensor_ctrl<N, double> ctrl(btc);

    addition_schedule<N, Traits> asch(m_sym, ctrl.req_const_symmetry());
    asch.build(m_sch, ctrl);

    bto_aux_add<N, Traits> out(m_sym, asch, btc, d);
    perform(out);
}


template<size_t N>
void btod_mult<N>::compute_block(bool zero, dense_tensor_i<N, double> &blk,
    const index<N> &idx, const tensor_transf<N, double> &tr,
    const double &c) {

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);

    permutation<N> pinva(m_pa, true), pinvb(m_pb, true), pinvc(tr.get_perm(), true);
    index<N> idxa(idx), idxb(idx);
    idxa.permute(pinva);
    idxb.permute(pinvb);

    orbit<N, double> oa(ctrla.req_const_symmetry(), idxa);
    abs_index<N> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &tra = oa.get_transf(idxa);

    orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
    abs_index<N> cidxb(ob.get_abs_canonical_index(),
            m_btb.get_bis().get_block_index_dims());
    const tensor_transf<N, double> &trb = ob.get_transf(idxb);

    permutation<N> pa(tra.get_perm());
    pa.permute(m_pa);
    pa.permute(pinvc);
    permutation<N> pb(trb.get_perm());
    pb.permute(m_pb);
    pb.permute(pinvc);

    dense_tensor_i<N, double> &blka = ctrla.req_block(cidxa.get_index());
    dense_tensor_i<N, double> &blkb = ctrlb.req_block(cidxb.get_index());

    double k = m_c * tr.get_scalar_tr().get_coeff() *
            tra.get_scalar_tr().get_coeff();
    if (m_recip)
        k /= trb.get_scalar_tr().get_coeff();
    else
        k *= trb.get_scalar_tr().get_coeff();

    if(zero) tod_set<N>().perform(blk);
    tod_mult<N>(blka, pa, blkb, pb, m_recip, k).perform(false, c, blk);

    ctrla.ret_block(cidxa.get_index());
    ctrlb.ret_block(cidxb.get_index());
}

template<size_t N>
void btod_mult<N>::make_schedule() {

    static const char *method = "make_schedule()";

    block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb);

    orbit_list<N, double> ol(m_sym);

    for (typename orbit_list<N, double>::iterator iol = ol.begin();
            iol != ol.end(); iol++) {

        index<N> idx(ol.get_index(iol));
        index<N> idxa(idx), idxb(idx);
        permutation<N> pinva(m_pa, true), pinvb(m_pb, true);
        idxa.permute(pinva);
        idxb.permute(pinvb);

        orbit<N, double> oa(ctrla.req_const_symmetry(), idxa);
        if (! oa.is_allowed())
            continue;
        abs_index<N> cidxa(oa.get_abs_canonical_index(),
                m_bta.get_bis().get_block_index_dims());
        bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());

        orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
        if (! ob.is_allowed()) {
            if (m_recip)
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Block not allowed in btb.");

            continue;
        }

        abs_index<N> cidxb(ob.get_abs_canonical_index(),
                m_btb.get_bis().get_block_index_dims());
        bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

        if (m_recip && zerob) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "zero in btb");
        }

        if (! zeroa && ! zerob) {
            m_sch.insert(idx);
        }

    }


}


template<size_t N, typename T>
btod_mult_task<N, T>::btod_mult_task(btod_mult<N> &bto,
    block_tensor_i<N, T> &btc, const index<N> &idx,
    bto_stream_i<N, btod_traits> &out) :

    m_bto(bto), m_btc(btc), m_idx(idx), m_out(out) {

}


template<size_t N, typename T>
void btod_mult_task<N, T>::perform() {

    block_tensor_ctrl<N, T> cc(m_btc);
    dense_tensor_i<N, T> &blk = cc.req_block(m_idx);
    tensor_transf<N, T> tr0;
    m_bto.compute_block(true, blk, m_idx, tr0, 1.0);
    m_out.put(m_idx, blk, tr0);
    cc.ret_block(m_idx);
    cc.req_zero_block(m_idx);
}


template<size_t N, typename T>
btod_mult_task_iterator<N, T>::btod_mult_task_iterator(btod_mult<N> &bto,
    block_tensor_i<N, T> &btc, bto_stream_i<N, btod_traits> &out) :

    m_bto(bto), m_btc(btc), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename T>
bool btod_mult_task_iterator<N, T>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename T>
libutil::task_i *btod_mult_task_iterator<N, T>::get_next() {

    dimensions<N> bidims = m_btc.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    btod_mult_task<N, T> *t =
        new btod_mult_task<N, T>(m_bto, m_btc, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename T>
void btod_mult_task_observer<N, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_IMPL_H
