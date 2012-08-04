#ifndef LIBTENSOR_BTOD_SYMMETRIZE_IMPL_H
#define LIBTENSOR_BTOD_SYMMETRIZE_IMPL_H

#include <list>
#include <libtensor/core/allocator.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_permute.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include "../../bto/impl/bto_aux_add_impl.h"
#include "../../bto/impl/bto_aux_symmetrize_impl.h"
#include "../btod_symmetrize.h"

namespace libtensor {


template<size_t N>
const char *btod_symmetrize<N>::k_clazz = "btod_symmetrize<N>";


template<size_t N>
btod_symmetrize<N>::btod_symmetrize(additive_bto<N, bto_traits<double> > &op,
        size_t i1, size_t i2, bool symm) :

    m_op(op), m_symm(symm), m_bis(op.get_bis()), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    static const char *method =
            "btod_symmetrize(additive_bto<N, bto_traits<double> >&, "
            "size_t, size_t, bool)";

    if(i1 == i2) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i");
    }
    m_perm1.permute(i1, i2);
    make_symmetry();
    make_schedule();
}


template<size_t N>
btod_symmetrize<N>::btod_symmetrize(additive_bto<N, bto_traits<double> > &op,
    const permutation<N> &perm, bool symm) :

    m_op(op), m_symm(symm), m_perm1(perm), m_bis(op.get_bis()),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    static const char *method = "btod_symmetrize(additive_bto<N, bto_traits<double> >&, "
        "const permutation<N>&, bool)";

    permutation<N> p1(perm); p1.permute(perm);
    if(perm.is_identity() || !p1.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm");
    }
    make_symmetry();
    make_schedule();
}


template<size_t N>
void btod_symmetrize<N>::sync_on() {

    m_op.sync_on();
}


template<size_t N>
void btod_symmetrize<N>::sync_off() {

    m_op.sync_off();
}


template<size_t N>
void btod_symmetrize<N>::perform1(block_tensor_i<N, double> &bt) {

    typedef bto_traits<double> Traits;

    block_tensor_ctrl<N, double> ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<N, double>(m_sym).perform(ctrl.req_symmetry());

    addition_schedule<N, Traits> asch(m_sym, m_sym);
    asch.build(m_sch, ctrl);

    bto_aux_add<N, Traits> out(m_sym, asch, bt, 1.0);
    perform1(out);
}


template<size_t N>
void btod_symmetrize<N>::perform1(block_tensor_i<N, double> &bt, double d) {

    typedef bto_traits<double> Traits;

    block_tensor_ctrl<N, double> ctrl(bt);

    addition_schedule<N, Traits> asch(m_sym, ctrl.req_const_symmetry());
    asch.build(m_sch, ctrl);

    bto_aux_add<N, Traits> out(m_sym, asch, bt, d);
    perform1(out);
}


template<size_t N>
void btod_symmetrize<N>::perform1(bto_stream_i< N, bto_traits<double> > &out) {

    typedef bto_traits<double> Traits;

    btod_symmetrize<N>::start_timer();

    try {

        bto_aux_symmetrize<N, Traits> out2(m_sym, out);
        // m_op.perform(out2);

    } catch(...) {
        btod_symmetrize<N>::stop_timer();
        throw;
    }

    btod_symmetrize<N>::stop_timer();
}


template<size_t N>
void btod_symmetrize<N>::compute_block(bool zero,
        dense_tensor_i<N, double> &blk, const index<N> &idx,
        const tensor_transf<N, double> &tr, const double &c) {

    typedef typename sym_schedule_t::iterator iterator_t;

    if(zero) tod_set<N>().perform(blk);

    dimensions<N> bidims(m_bis.get_block_index_dims());
    abs_index<N> aidx(idx, bidims);

    std::list<schrec> sch1;
    std::pair<iterator_t, iterator_t> jr =
        m_sym_sch.equal_range(aidx.get_abs_index());
    for(iterator_t j = jr.first; j != jr.second; ++j) {
        sch1.push_back(j->second);
    }

    while(!sch1.empty()) {
        abs_index<N> ai(sch1.front().ai, bidims);
        size_t n = 0;
        for(typename std::list<schrec>::iterator j = sch1.begin();
            j != sch1.end(); ++j) {
            if(j->ai == ai.get_abs_index()) n++;
        }

        tensor_transf<N, double> tri(sch1.front().tr);
        tri.transform(tr);

        if(n == 1) {
            additive_bto<N, bto_traits<double> >::compute_block(m_op, false,
                blk, ai.get_index(), tri, c);
            sch1.pop_front();
        } else {
            dimensions<N> dims(blk.get_dims());
            // TODO: replace with "temporary block" feature
            dense_tensor< N, double, allocator<double> > tmp(dims);
            additive_bto<N, bto_traits<double> >::compute_block(m_op, true,
                tmp, ai.get_index(), tri, c);
            tensor_transf<N, double> tri_inv(tri);
            tri_inv.invert();
            for(typename std::list<schrec>::iterator j =
                sch1.begin(); j != sch1.end();) {
                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, double> trj(tri_inv);
                trj.transform(j->tr);
                trj.transform(tr);
                tod_copy<N>(tmp, trj.get_perm(),
                    trj.get_scalar_tr().get_coeff()).perform(false, 1.0, blk);
                j = sch1.erase(j);
            }
        }
    }
}


template<size_t N>
void btod_symmetrize<N>::make_symmetry() {

    sequence<N, size_t> seq2(0), idxgrp(0), symidx(0);
    for (register size_t i = 0; i < N; i++) seq2[i] = i;
    m_perm1.apply(seq2);

    size_t idx = 1;
    for (register size_t i = 0; i < N; i++) {
        if (seq2[i] <= i) continue;

        idxgrp[i] = 1;
        idxgrp[seq2[i]] = 2;
        symidx[i] = symidx[seq2[i]] = idx++;
    }
    scalar_transf<double> tr(m_symm ? 1. : -1.);
    so_symmetrize<N, double>(m_op.get_symmetry(),
            idxgrp, symidx, tr, tr).perform(m_sym);
}


template<size_t N>
void btod_symmetrize<N>::make_schedule() {

    btod_symmetrize<N>::start_timer("make_schedule");

    dimensions<N> bidims(m_bis.get_block_index_dims());
    orbit_list<N, double> ol(m_sym);

    const assignment_schedule<N, double> &sch0 = m_op.get_schedule();
    for(typename assignment_schedule<N, double>::iterator i = sch0.begin();
        i != sch0.end(); i++) {

        abs_index<N> ai0(sch0.get_abs_index(i), bidims);
        orbit<N, double> o(m_op.get_symmetry(), ai0.get_index());

        for(typename orbit<N, double>::iterator j = o.begin();
            j != o.end(); j++) {

            abs_index<N> aj1(o.get_abs_index(j), bidims);
            index<N> j2(aj1.get_index()); j2.permute(m_perm1);
            abs_index<N> aj2(j2, bidims);

            if(ol.contains(aj1.get_abs_index())) {
                if(!m_sch.contains(aj1.get_abs_index())) {
                    m_sch.insert(aj1.get_abs_index());
                }
                tensor_transf<N, double> tr1(o.get_transf(j));
                m_sym_sch.insert(sym_schedule_pair_t(
                    aj1.get_abs_index(),
                    schrec(ai0.get_abs_index(), tr1)));
            }
            if(ol.contains(aj2.get_abs_index())) {
                if(!m_sch.contains(aj2.get_abs_index())) {
                    m_sch.insert(aj2.get_abs_index());
                }
                tensor_transf<N, double> tr2(o.get_transf(j));
                tr2.permute(m_perm1);
                tr2.transform(scalar_transf<double>(m_symm ? 1.0 : -1.0));
                m_sym_sch.insert(sym_schedule_pair_t(
                    aj2.get_abs_index(),
                    schrec(ai0.get_abs_index(), tr2)));
            }
        }
    }

    btod_symmetrize<N>::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE_IMPL_H
