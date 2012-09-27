#ifndef LIBTENSOR_BTOD_SUM_IMPL_H
#define LIBTENSOR_BTOD_SUM_IMPL_H

#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include "../btod_sum.h"

namespace libtensor {


template<size_t N>
const char* btod_sum<N>::k_clazz = "btod_sum<N>";


template<size_t N>
inline btod_sum<N>::btod_sum(additive_bto<N, btod_traits> &op, double c) :
    m_bis(op.get_bis()), m_bidims(m_bis.get_block_index_dims()),
    m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

    so_copy<N, double>(op.get_symmetry()).perform(m_sym);
    add_op(op, c);
}


template<size_t N>
btod_sum<N>::~btod_sum() {

    delete m_sch;
}


template<size_t N>
void btod_sum<N>::perform(bto_stream_i<N, btod_traits> &out) {

    if(m_ops.empty()) return;

    block_tensor< N, double, allocator<double> > bt(m_bis);

    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); ++iop) {

        if(iop == m_ops.begin()) {
            block_tensor_ctrl<N, double> ctrl(bt);
            so_copy<N, double>(iop->get_op().get_symmetry()).
                perform(ctrl.req_symmetry());
        }
        iop->get_op().perform(bt, iop->get_coeff());
    }

    btod_copy<N>(bt).perform(out);
}


template<size_t N>
void btod_sum<N>::compute_block(bool zero, dense_tensor_i<N, double> &blk,
    const index<N> &i, const tensor_transf<N, double> &tr, const double &c) {

    if(zero) tod_set<N>().perform(blk);

    abs_index<N> ai(i, m_bidims);

    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {

        if(iop->get_op().get_schedule().contains(ai.get_abs_index())) {
            additive_bto<N, btod_traits>::compute_block(iop->get_op(),
                false, blk, i, tr, c * iop->get_coeff());
        } else {
            const symmetry<N, double> &sym = iop->get_op().get_symmetry();
            orbit<N, double> orb(sym, i);
            if(!orb.is_allowed()) continue;
            abs_index<N> ci(orb.get_abs_canonical_index(), m_bidims);

            if(iop->get_op().get_schedule().contains(ci.get_abs_index())) {
                tensor_transf<N, double> tra(orb.get_transf(i));
                tra.transform(tr);

                additive_bto<N, btod_traits>::compute_block(
                    iop->get_op(), false, blk, ci.get_index(), tra,
                    c * iop->get_coeff());
            }
        }
    }
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &btb) {

/*
    bool first = true;
    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {

        if(first) {
            iop->get_op().perform(bt);
            if(iop->get_coeff() != 1.0) {
                btod_scale<N>(bt, iop->get_coeff()).perform();
            }
            first = false;
        } else {
            iop->get_op().perform(bt, iop->get_coeff());
        }
    }
 */
    bto_aux_copy<N, btod_traits> out(m_sym, btb);
    perform(out);
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &btb, const double &c) {

/*
    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {

        iop->get_op().perform(bt, c * iop->get_coeff());
    }
 */

    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, btod_traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N, btod_traits> out(m_sym, asch, btb, c);
    perform(out);
}


template<size_t N>
void btod_sum<N>::add_op(additive_bto<N, btod_traits> &op, double c) {

    static const char *method =
            "add_op(additive_bto<N, btod_traits>&, double)";

    block_index_space<N> bis(m_bis), bis1(op.get_bis());
    bis.match_splits();
    bis1.match_splits();
    if(!bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "op");
    }
    if(c == 0.0) return;

    if(m_ops.empty()) {
        so_copy<N, double>(op.get_symmetry()).perform(m_sym);
    } else {
        permutation<N + N> perm0;
        block_index_space_product_builder<N, N> bbx(m_bis, m_bis, perm0);

        symmetry<N + N, double> symx(bbx.get_bis());
        so_dirsum<N, N, double>(m_sym, op.get_symmetry(), perm0).perform(symx);
        mask<N + N> msk;
        sequence<N + N, size_t> seq;
        for (register size_t i = 0; i < N; i++) {
            msk[i] = msk[i + N] = true;
            seq[i] = seq[i + N] = i;
        }
        so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);
    }
    m_ops.push_back(node_t(op, c));
    m_dirty_sch = true;
}


template<size_t N>
void btod_sum<N>::make_schedule() const {

    delete m_sch;
    m_sch = new assignment_schedule<N, double>(m_bidims);

    orbit_list<N, double> ol(m_sym);
    std::list< orbit_list<N, double>* > op_ol;
    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {
        op_ol.push_back(new orbit_list<N, double>(
            iop->get_op().get_symmetry()));
    }

    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); io++) {

        bool zero = true;
        typename std::list< orbit_list<N, double>* >::iterator iol =
            op_ol.begin();
        for(typename std::list<node_t>::iterator iop = m_ops.begin();
            zero && iop != m_ops.end(); iop++) {

            if(!(*iol)->contains(ol.get_abs_index(io))) {
                orbit<N, double> o(iop->get_op().get_symmetry(),
                    ol.get_index(io));
                if(!o.is_allowed()) continue;
                if(iop->get_op().get_schedule().contains(
                    o.get_abs_canonical_index())) {
                    zero = false;
                }
            } else {
                if(iop->get_op().get_schedule().contains(
                    ol.get_abs_index(io))) {
                    zero = false;
                }
            }
            iol++;
        }

        if(!zero) m_sch->insert(ol.get_abs_index(io));
    }

    for(typename std::list< orbit_list<N, double>* >::iterator i =
        op_ol.begin(); i != op_ol.end(); i++) delete *i;

    m_dirty_sch = false;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_IMPL_H
