#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H

#include <algorithm> // for std::swap
#include <libtensor/core/allocator.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_symmetrize.h>
#include "../btod_symmetrize3.h"

namespace libtensor {


template<size_t N>
const char *btod_symmetrize3<N>::k_clazz = "btod_symmetrize3<N>";


template<size_t N>
btod_symmetrize3<N>::btod_symmetrize3(additive_gen_bto<N, bti_traits> &op,
    size_t i1, size_t i2, size_t i3, bool symm) :

    m_op(op), m_i1(i1), m_i2(i2), m_i3(i3), m_symm(symm),
    m_sym(op.get_bis()),
    m_sch(op.get_bis().get_block_index_dims()) {

    static const char *method =
        "btod_symmetrize3(additive_bto<N, btod_traits>&, size_t, size_t, "
        "size_t, bool)";

    if(i1 == i2 || i2 == i3 || i1 == i3) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i1,i2,i3");
    }

    if(m_i1 > m_i2) std::swap(m_i1, m_i2);
    if(m_i2 > m_i3) std::swap(m_i2, m_i3);
    if(m_i1 > m_i2) std::swap(m_i1, m_i2);

    make_symmetry();
    make_schedule();
}


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_stream_i<N, bti_traits> &out) {

    typedef btod_traits Traits;

    try {

        permutation<N> perm1, perm2, perm3;
        perm1.permute(m_i1, m_i2);
        perm2.permute(m_i1, m_i3);
        perm3.permute(m_i2, m_i3);

        scalar_transf<double> str(m_symm ? 1.0 : -1.0);

        tensor_transf<N, double> tr0;
        tensor_transf<N, double> tr1(perm1, str);
        tensor_transf<N, double> tr2(perm2, str);
        tensor_transf<N, double> tr3(perm3, str);
        tensor_transf<N, double> tr4(tr1), tr5(tr1);
        tr4.transform(tr2);
        tr5.transform(tr3);

        gen_bto_aux_symmetrize<N, Traits> out2(m_op.get_symmetry(), m_sym, out);
        out2.add_transf(tr0);
        out2.add_transf(tr1);
        out2.add_transf(tr2);
        out2.add_transf(tr3);
        out2.add_transf(tr4);
        out2.add_transf(tr5);
        m_op.perform(out2);

    } catch(...) {
        throw;
    }
}


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_tensor_i<N, bti_traits> &bt) {

    typedef btod_traits Traits;
    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<N, double>(m_sym).perform(ctrl.req_symmetry());

    addition_schedule<N, Traits> asch(m_sym, m_sym);
    asch.build(m_sch, ctrl);

    gen_bto_aux_add<N, Traits> out(m_sym, asch, bt, scalar_transf<double>());
    perform(out);
}


template<size_t N>
void btod_symmetrize3<N>::perform(gen_block_tensor_i<N, bti_traits> &bt,
    const scalar_transf<double> &d) {

    typedef btod_traits Traits;
    typedef typename btod_traits::bti_traits bti_traits;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    addition_schedule<N, Traits> asch(m_sym, ctrl.req_const_symmetry());
    asch.build(m_sch, ctrl);

    gen_bto_aux_add<N, Traits> out(m_sym, asch, bt, d);
    perform(out);
}


template<size_t N>
void btod_symmetrize3<N>::perform(block_tensor_i<N, double> &bt, double d) {

    perform(bt, scalar_transf<double>(d));
}


template<size_t N>
void btod_symmetrize3<N>::compute_block(
        bool zero,
        const index<N> &i,
        const tensor_transf<N, double> &tr,
        dense_tensor_wr_i<N, double> &blk) {

    typedef typename sym_schedule_t::iterator iterator_t;

    if(zero) tod_set<N>().perform(blk);

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    abs_index<N> ai(i, bidims);

    sym_schedule_t sch;
    make_schedule_blk(ai, sch);

    std::pair<iterator_t, iterator_t> jr =
        sch.equal_range(ai.get_abs_index());
    std::list<schrec> sch1;
    for(iterator_t j = jr.first; j != jr.second; ++j) {
        sch1.push_back(j->second);
    }
    sch.clear();

    while(!sch1.empty()) {
        abs_index<N> ai(sch1.front().ai, bidims);
        size_t n = 0;
        for(typename std::list<schrec>::iterator j = sch1.begin();
            j != sch1.end(); ++j) {
            if(j->ai == ai.get_abs_index()) n++;
        }
        if(n == 1) {
            tensor_transf<N, double> tri(sch1.front().tr);
            tri.transform(tr);
            additive_gen_bto<N, bti_traits>::compute_block(m_op, false,
                    ai.get_index(), tri, blk);
            sch1.pop_front();
        } else {
            dimensions<N> dims(blk.get_dims());
            dims.permute(permutation<N>(tr.get_perm(), true));
            dims.permute(permutation<N>(sch1.front().tr.get_perm(),
                true));
            // TODO: replace with "temporary block" feature
            dense_tensor< N, double, allocator<double> > tmp(dims);
            additive_gen_bto<N, bti_traits>::compute_block(m_op, true,
                    ai.get_index(), tensor_transf<N, double>(), tmp);
            for(typename std::list<schrec>::iterator j =
                sch1.begin(); j != sch1.end();) {

                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, double> trj(j->tr);
                trj.transform(tr);
                tod_copy<N>(tmp, trj).perform(false, blk);
                j = sch1.erase(j);
            }
        }
    }
}


template<size_t N>
void btod_symmetrize3<N>::make_symmetry() {

    sequence<N, size_t> seq1, seq2;
    seq1[m_i1] = 1; seq1[m_i2] = 2; seq1[m_i3] = 3;
    seq2[m_i1] = seq2[m_i2] = seq2[m_i3] = 1;
    scalar_transf<double> tr0, tr1(-1.);
    so_symmetrize<N, double>(m_op.get_symmetry(),
            seq1, seq2, m_symm ? tr0 : tr1, tr0).perform(m_sym);

}


template<size_t N>
void btod_symmetrize3<N>::make_schedule() {

    btod_symmetrize3<N>::start_timer("make_schedule");

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    orbit_list<N, double> ol(m_sym);

    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); io++) {

        abs_index<N> ai(ol.get_index(io), bidims);
        sym_schedule_t sch;
        make_schedule_blk(ai, sch);
        if(!sch.empty()) m_sch.insert(ai.get_abs_index());
    }

    btod_symmetrize3<N>::stop_timer("make_schedule");
}


template<size_t N>
void btod_symmetrize3<N>::make_schedule_blk(const abs_index<N> &ai,
    sym_schedule_t &sch) const {

    permutation<N> perm1, perm2, perm3;
    perm1.permute(m_i1, m_i2);
    perm2.permute(m_i1, m_i3);
    perm3.permute(m_i2, m_i3);
    double scal = m_symm ? 1.0 : -1.0;

    index<N> idx0(ai.get_index()), idx1(idx0), idx2(idx0), idx3(idx0),
        idx4(idx0), idx5(idx0);
    idx1.permute(perm1);
    idx2.permute(perm2);
    idx3.permute(perm3);
    idx4.permute(perm1).permute(perm2);
    idx5.permute(perm1).permute(perm3);

    const symmetry<N, double> &sym0 = m_op.get_symmetry();
    const assignment_schedule<N, double> &sch0 = m_op.get_schedule();

    orbit<N, double> o0(sym0, idx0), o1(sym0, idx1), o2(sym0, idx2),
        o3(sym0, idx3), o4(sym0, idx4), o5(sym0, idx5);

    //  This is a temporary schedule for the formation of the block
    std::list<schrec> sch1;

    //  Form the temporary schedule

    if(sch0.contains(o0.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o0.get_transf(idx0));
        sch1.push_back(schrec(o0.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o1.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o1.get_transf(idx1));
        tr.permute(perm1);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o1.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o2.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o2.get_transf(idx2));
        tr.permute(perm2);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o2.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o3.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o3.get_transf(idx3));
        tr.permute(perm3);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o3.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o4.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o4.get_transf(idx4));
        tr.permute(perm1);
        tr.permute(perm3);
        sch1.push_back(schrec(o4.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o5.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o5.get_transf(idx5));
        tr.permute(perm1);
        tr.permute(perm2);
        sch1.push_back(schrec(o5.get_abs_canonical_index(), tr));
    }

    //  Consolidate and transfer the temporary schedule

    while(!sch1.empty()) {

        typename std::list<schrec>::iterator i = sch1.begin();
        abs_index<N> aidx(i->ai, ai.get_dims());
        double c = 0.0;
        tensor_transf<N, double> tr0(i->tr);

        do {
            if(i->ai != aidx.get_abs_index()) {
                ++i;
                continue;
            }
            if(tr0.get_perm().equals(i->tr.get_perm())) {
                c += i->tr.get_scalar_tr().get_coeff();
                i = sch1.erase(i);
                continue;
            }
            ++i;
        } while(i != sch1.end());
        if(c != 0.0) {
            tensor_transf<N, double> tr;
            tr.permute(tr0.get_perm());
            tr.transform(scalar_transf<double>(c));
            sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
                schrec(aidx.get_abs_index(), tr)));
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_IMPL_H
