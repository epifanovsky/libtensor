#ifndef LIBTENSOR_BTOD_ADD_IMPL_H
#define LIBTENSOR_BTOD_ADD_IMPL_H

#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_permute.h>
#include <libtensor/dense_tensor/tod_add.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../btod_add.h"

namespace libtensor {


template<size_t N>
const char *btod_add<N>::k_clazz = "btod_add<N>";


template<size_t N, typename Traits>
class bto_add_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    btod_add<N> &m_bto;
    block_tensor_type &m_btb;
    index<N> m_idx;
    bto_stream_i<N, Traits> &m_out;

public:
    bto_add_task(
        btod_add<N> &bto,
        block_tensor_type &btb,
        const index<N> &idx,
        bto_stream_i<N, Traits> &out);

    virtual ~bto_add_task() { }
    virtual void perform();

};


template<size_t N, typename Traits>
class bto_add_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    btod_add<N> &m_bto;
    block_tensor_type &m_btb;
    bto_stream_i<N, Traits> &m_out;
    const assignment_schedule<N, element_type> &m_sch;
    typename assignment_schedule<N, element_type>::iterator m_i;

public:
    bto_add_task_iterator(
        btod_add<N> &bto,
        block_tensor_type &btb,
        bto_stream_i<N, Traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class bto_add_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, double c) :

    m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims()),
    m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

    m_bis.match_splits();
    add_operand(bt, permutation<N>(), c);
}


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, const permutation<N> &perm,
        double c) :

        m_bis(block_index_space<N>(bt.get_bis()).permute(perm)),
        m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_dirty_sch(true),
        m_sch(0) {

    m_bis.match_splits();
    add_operand(bt, perm, c);
}


template<size_t N>
btod_add<N>::~btod_add() {

    delete m_sch;

    typename std::vector<operand_t*>::iterator i = m_ops.begin();
    while(i != m_ops.end()) {
        delete (*i);
        *i = NULL;
        i++;
    }
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt, double c) {

    static const char *method =
            "add_op(block_tensor_i<N, double>&, double)";

    if(fabs(c) == 0.0) return;

    block_index_space<N> bis(bt.get_bis());
    bis.match_splits();
    if(!m_bis.equals(bis)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
                __LINE__, "bt");
    }

    add_operand(bt, permutation<N>(), c);
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt,
        const permutation<N> &perm, double c) {

    static const char *method = "add_op(block_tensor_i<N, double>&, "
            "const permutation<N>&, double)";

    if(fabs(c) == 0.0) return;

    block_index_space<N> bis(bt.get_bis());
    bis.match_splits();
    bis.permute(perm);
    if(!m_bis.equals(bis)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
                __LINE__, "bt");
    }

    add_operand(bt, perm, c);
}


template<size_t N>
void btod_add<N>::sync_on() {

    size_t narg = m_ops.size();
    for(size_t i = 0; i < narg; i++) {
        block_tensor_ctrl<N, double> ctrl(m_ops[i]->m_bt);
        ctrl.req_sync_on();
    }
}


template<size_t N>
void btod_add<N>::sync_off() {

    size_t narg = m_ops.size();
    for(size_t i = 0; i < narg; i++) {
        block_tensor_ctrl<N, double> ctrl(m_ops[i]->m_bt);
        ctrl.req_sync_off();
    }
}


template<size_t N>
void btod_add<N>::perform(bto_stream_i<N, btod_traits> &out) {

    typedef btod_traits Traits;
    typedef double element_t;
    typedef allocator<element_t> allocator_type;

    try {

        out.open();

        // TODO: replace with temporary block tensor from traits
        block_tensor<N, element_t, allocator_type> btb(m_bis);
        block_tensor_ctrl<N, element_t> cb(btb);
        cb.req_sync_on();
        sync_on();

        bto_add_task_iterator<N, Traits> ti(*this, btb, out);
        bto_add_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

        cb.req_sync_off();
        sync_off();

        out.close();

    } catch(...) {
        throw;
    }
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &btb) {

    typedef btod_traits Traits;

    bto_aux_copy<N, Traits> out(m_sym, btb);
    perform(out);
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &btb, const double &c) {

    typedef btod_traits Traits;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    block_tensor_ctrl_type cb(btb);
    addition_schedule<N, Traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N, Traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N>
void btod_add<N>::compute_block(bool zero, dense_tensor_i<N, double> &blkb,
    const index<N> &ib, const tensor_transf<N, double> &trb, const double &kb) {

    static const char *method = "compute_block(bool, tensor_i<N, double>&, "
        "const index<N>&, const tensor_transf<N, double>&, double)";

    btod_add<N>::start_timer();

    try {

        abs_index<N> aib(ib, m_bidims);
        std::pair<schiterator_t, schiterator_t> ipair =
                m_op_sch.equal_range(aib.get_abs_index());
        if(ipair.first != m_op_sch.end()) {
            compute_block(blkb, ipair, zero, trb, kb);
        }

    } catch(...) {
        btod_add<N>::stop_timer();
        throw;
    }

    btod_add<N>::stop_timer();
}


template<size_t N>
void btod_add<N>::compute_block(dense_tensor_i<N, double> &blkb,
    const std::pair<schiterator_t, schiterator_t> ipair, bool zero,
    const tensor_transf<N, double> &trb, double kb) {

    size_t narg = m_ops.size();
    std::vector<block_tensor_ctrl<N, double>*> ca(narg);
    for(size_t i = 0; i < narg; i++) {
        ca[i] = new block_tensor_ctrl<N, double>(m_ops[i]->m_bt);
    }

    schiterator_t iarg = ipair.first;
    tod_add<N> *op = 0;

    {
        const schrec &rec = iarg->second;
        permutation<N> perm(rec.perm); perm.permute(trb.get_perm());
        double k = rec.k * kb * trb.get_scalar_tr().get_coeff();
        op = new tod_add<N>(ca[rec.iarg]->req_block(rec.idx), perm, k);
    }

    for(iarg++; iarg != ipair.second; iarg++) {
        const schrec &rec = iarg->second;
        permutation<N> perm(rec.perm); perm.permute(trb.get_perm());
        double k = rec.k * kb * trb.get_scalar_tr().get_coeff();
        op->add_op(ca[rec.iarg]->req_block(rec.idx), perm, k);
    }

    op->perform(zero, 1.0, blkb);

    delete op;

    for(iarg = ipair.first; iarg != ipair.second; iarg++) {
        const schrec &rec = iarg->second;
        ca[rec.iarg]->ret_block(rec.idx);
    }

    for(size_t i = 0; i < narg; i++) delete ca[i];
}


template<size_t N>
void btod_add<N>::add_operand(block_tensor_i<N, double> &bt,
    const permutation<N> &perm, double c) {

    static const char *method = "add_operand(block_tensor_i<N,double>&, "
        "const permutation<N>&, double)";

    bool first = m_ops.empty();

    try {
        m_ops.push_back(new operand(bt, perm, c));
    } catch(std::bad_alloc &e) {
        throw out_of_memory("libtensor", k_clazz, method, __FILE__,
                __LINE__, "op");
    }

    block_tensor_ctrl<N, double> ca(bt);
    if(first) {
        so_permute<N, double>(ca.req_const_symmetry(), perm).perform(m_sym);
    } else {

        sequence<N, size_t> seq2a;
        sequence<N + N, size_t> seq1b, seq2b;
        for (size_t i = 0; i < N; i++) {
            seq2a[i] = i + N;
            seq1b[i] = seq2b[i] = i;
        }
        perm.apply(seq2a);
        for (size_t i = N; i < N + N; i++) {
            seq1b[i] = i; seq2b[i] = seq2a[i - N];
        }
        permutation_builder<N + N> pb(seq2b, seq1b);

        block_index_space_product_builder<N, N> bbx(m_bis, m_bis,
            pb.get_perm());

        symmetry<N + N, double> symx(bbx.get_bis());
        so_dirsum<N, N, double>(m_sym, ca.req_const_symmetry(),
            pb.get_perm()).perform(symx);
        mask<N + N> msk;
        sequence<N + N, size_t> seq;
        for (register size_t i = 0; i < N; i++) {
            msk[i] = msk[seq2a[i]] = true;
            seq[i] = seq[seq2a[i]] = i;
        }
        so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);
    }
    m_dirty_sch = true;
}


template<size_t N>
void btod_add<N>::make_schedule() const {

    //~ btod_add<N>::start_timer("make_schedule");

    delete m_sch;
    m_sch = new assignment_schedule<N, double>(m_bidims);
    m_op_sch.clear();

    size_t narg = m_ops.size();
    std::vector<block_tensor_ctrl<N, double>*> ca(narg);
    std::vector<orbit_list<N, double>*> ola(narg);

    for(size_t i = 0; i < narg; i++) {
        ca[i] = new block_tensor_ctrl<N, double>(m_ops[i]->m_bt);
        ola[i] = new orbit_list<N, double>(ca[i]->req_const_symmetry());
    }

    orbit_list<N, double> olb(m_sym);
    for(typename orbit_list<N, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        size_t nrec = 0;

        for(size_t i = 0; i < narg; i++) {

            permutation<N> pinv(m_ops[i]->m_perm, true);
            index<N> ia(olb.get_index(iob)); ia.permute(pinv);
            dimensions<N> bidimsa(m_bidims); bidimsa.permute(pinv);
            abs_index<N> aia(ia, bidimsa);

            if(!ola[i]->contains(aia.get_abs_index())) {

                orbit<N, double> oa(ca[i]->req_const_symmetry(),
                        ia);
                if(!oa.is_allowed()) continue;
                abs_index<N> acia(oa.get_abs_canonical_index(),
                        bidimsa);

                if(ca[i]->req_is_zero_block(acia.get_index()))
                    continue;

                const tensor_transf<N, double> &tra = oa.get_transf(
                        aia.get_abs_index());

                schrec rec;
                rec.iarg = i;
                rec.idx = acia.get_index();
                rec.k = m_ops[i]->m_c * tra.get_scalar_tr().get_coeff();
                rec.perm.permute(tra.get_perm()).
                        permute(m_ops[i]->m_perm);
                m_op_sch.insert(std::pair<size_t, schrec>(
                        olb.get_abs_index(iob), rec));
                nrec++;
            } else {

                if(ca[i]->req_is_zero_block(ia)) continue;

                schrec rec;
                rec.iarg = i;
                rec.idx = aia.get_index();
                rec.k = m_ops[i]->m_c;
                rec.perm.permute(m_ops[i]->m_perm);
                m_op_sch.insert(std::pair<size_t, schrec>(
                        olb.get_abs_index(iob), rec));
                nrec++;
            }
        }

        if(nrec > 0) m_sch->insert(olb.get_abs_index(iob));
    }

    for(size_t i = 0; i < narg; i++) {
        delete ola[i];
        delete ca[i];
    }

    m_dirty_sch = false;

    //~ btod_add<N>::stop_timer("make_schedule");
}


template<size_t N, typename Traits>
bto_add_task<N, Traits>::bto_add_task(btod_add<N> &bto,
    block_tensor_type &btb, const index<N> &idx,
    bto_stream_i<N, Traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, typename Traits>
void bto_add_task<N, Traits>::perform() {

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


template<size_t N, typename Traits>
bto_add_task_iterator<N, Traits>::bto_add_task_iterator(
    btod_add<N> &bto, block_tensor_type &btb,
    bto_stream_i<N, Traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename Traits>
bool bto_add_task_iterator<N, Traits>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename Traits>
libutil::task_i *bto_add_task_iterator<N, Traits>::get_next() {

    dimensions<N> bidims = m_btb.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    bto_add_task<N, Traits> *t =
        new bto_add_task<N, Traits>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void bto_add_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_IMPL_H
