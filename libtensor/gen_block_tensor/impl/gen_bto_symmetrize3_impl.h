#ifndef LIBTENSOR_GEN_BTO_SYMMETRIZE3_IMPL_H
#define LIBTENSOR_GEN_BTO_SYMMETRIZE3_IMPL_H

#include <set>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_symmetrize.h>
#include "../gen_bto_symmetrize3.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_symmetrize3<N, Traits, Timed>::k_clazz[] =
    "gen_bto_symmetrize3<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_symmetrize3<N, Traits, Timed>::gen_bto_symmetrize3(
    additive_gen_bto<N, bti_traits> &op,
    const permutation<N> &perm1,
    const permutation<N> &perm2,
    bool symm) :

    m_op(op), m_perm1(perm1), m_perm2(perm2), m_symm(symm),
    m_sym(op.get_bis()), m_sch(0) {

    static const char method[] =
        "gen_bto_symmetrize3(additive_bto<N, btod_traits>&, "
        "const permutation<N>&, const permutation<N>&, bool)";

    permutation<N> p1(perm1); p1.permute(perm1);
    if(perm1.is_identity() || !p1.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm1");
    }
    permutation<N> p2(perm2); p2.permute(perm2);
    if(perm2.is_identity() || !p2.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm2");
    }
    permutation<N> p12(perm1); p12.permute(perm2);
    permutation<N> p123(p12); p123.permute(p12).permute(p12);
    if(p12.is_identity() || !p123.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "perm1,perm2");
    }

    make_symmetry();
    make_schedule();
}


template<size_t N, typename Traits, typename Timed>
gen_bto_symmetrize3<N, Traits, Timed>::~gen_bto_symmetrize3() {

    delete m_sch;
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize3<N, Traits, Timed>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    try {

        scalar_transf<element_type> str(m_symm ? 1.0 : -1.0);

        tensor_transf<N, element_type> tr0;
        tensor_transf<N, element_type> tr1(m_perm1, str);
        tensor_transf<N, element_type> tr2(m_perm2, str);
        tensor_transf<N, element_type> tr3(tr1); tr3.transform(tr2);
        tensor_transf<N, element_type> tr4(tr2); tr4.transform(tr1);
        tensor_transf<N, element_type> tr5(tr3); tr5.transform(tr1);

        gen_bto_aux_symmetrize<N, Traits> out2(m_op.get_symmetry(), m_sym, out);
        out2.add_transf(tr0);
        out2.add_transf(tr1);
        out2.add_transf(tr2);
        out2.add_transf(tr3);
        out2.add_transf(tr4);
        out2.add_transf(tr5);
        out2.open();
        m_op.perform(out2);
        out2.close();

    } catch(...) {
        throw;
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize3<N, Traits, Timed>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    typedef typename Traits::template temp_block_type<N>::type temp_block_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy;
    typedef typename Traits::template to_set_type<N>::type to_set;

    typedef typename sym_schedule_type::iterator iterator_t;

    bool zero1 = zero;

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    abs_index<N> aib(ib, bidims);

    sym_schedule_type sch;
    make_schedule_blk(aib, sch);

    std::pair<iterator_t, iterator_t> jr = sch.equal_range(aib.get_abs_index());
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

            tensor_transf<N, element_type> tri(sch1.front().tr);
            tri.transform(trb);
            m_op.compute_block(zero1, ai.get_index(), tri, blkb);
            zero1 = false;
            sch1.pop_front();

        } else {

            dimensions<N> dims(blkb.get_dims());
            dims.permute(permutation<N>(trb.get_perm(), true));
            dims.permute(permutation<N>(sch1.front().tr.get_perm(), true));
            temp_block_type tblk(dims);
            m_op.compute_block(true, ai.get_index(),
                tensor_transf<N, element_type>(), tblk);
            for(typename std::list<schrec>::iterator j = sch1.begin();
                j != sch1.end();) {

                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, element_type> trj(j->tr);
                trj.transform(trb);
                to_copy(tblk, trj).perform(zero1, blkb);
                zero1 = false;
                j = sch1.erase(j);
            }

        }
    }

    if(zero1) to_set().perform(blkb);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize3<N, Traits, Timed>::make_symmetry() {

    sequence<N, size_t> seq1(0), seq2(0);
    for(size_t i = 0; i < N; i++) seq1[i] = seq2[i] = i;
    m_perm1.apply(seq1);
    m_perm2.apply(seq2);

    sequence<N, size_t> idxgrp(0), symidx(0);
    size_t i1 = 1, i2 = 1, i3 = 1;
    for(size_t i = 0; i < N; i++) {
        if(seq1[i] != i) {
            if(seq2[i] != i) {
                idxgrp[i] = 3;
                symidx[i] = i3++;
            } else {
                idxgrp[i] = 1;
                symidx[i] = i1++;
            }
        } else {
            if(seq2[i] != i) {
                idxgrp[i] = 2;
                symidx[i] = i2++;
            }
        }
    }
    scalar_transf<element_type> tr0, tr1(-1.0);
    so_symmetrize<N, double>(m_op.get_symmetry(), idxgrp, symidx,
        m_symm ? tr0 : tr1, tr0).perform(m_sym);
}


namespace {

template<size_t N, typename T>
void visit_orbit(const orbit<N, T> &o, std::set<size_t> &visited) {

    for(typename orbit<N, T>::iterator j = o.begin(); j != o.end(); j++) {
        visited.insert(o.get_abs_index(j));
    }
}

} // unnamed namespace


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize3<N, Traits, Timed>::make_schedule() const {

    delete m_sch;
    m_sch = new assignment_schedule<N, element_type>(
        m_op.get_bis().get_block_index_dims());

    gen_bto_symmetrize3::start_timer("make_schedule");

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    scalar_transf<element_type> scal(m_symm ? 1.0 : -1.0);

    std::set<size_t> visited;

    const assignment_schedule<N, element_type> &sch0 = m_op.get_schedule();
    for(typename assignment_schedule<N, element_type>::iterator i =
        sch0.begin(); i != sch0.end(); ++i) {

        abs_index<N> ai0(sch0.get_abs_index(i), bidims);
        orbit<N, element_type> o(m_op.get_symmetry(), ai0.get_index());

        for(typename orbit<N, element_type>::iterator j = o.begin();
            j != o.end(); j++) {

            abs_index<N> aj1(o.get_abs_index(j), bidims);
            if(visited.count(aj1.get_abs_index()) == 0) {
                orbit<N, element_type> o1(m_sym, aj1.get_abs_index());
                if(!m_sch->contains(o1.get_acindex())) {
                    m_sch->insert(o1.get_acindex());
                    visit_orbit(o1, visited);
                }
            }

            index<N> j2(aj1.get_index());
            j2.permute(m_perm1);
            abs_index<N> aj2(j2, bidims);
            if(visited.count(aj2.get_abs_index()) == 0) {
                orbit<N, element_type> o2(m_sym, aj2.get_abs_index());
                if(!m_sch->contains(o2.get_acindex())) {
                    m_sch->insert(o2.get_acindex());
                    visit_orbit(o2, visited);
                }
            }

            index<N> j3(aj1.get_index());
            j3.permute(m_perm2);
            abs_index<N> aj3(j3, bidims);
            if(visited.count(aj3.get_abs_index()) == 0) {
                orbit<N, element_type> o3(m_sym, aj3.get_abs_index());
                if(!m_sch->contains(o3.get_acindex())) {
                    m_sch->insert(o3.get_acindex());
                    visit_orbit(o3, visited);
                }
            }

            index<N> j4(aj1.get_index());
            j4.permute(m_perm1).permute(m_perm2);
            abs_index<N> aj4(j4, bidims);
            if(visited.count(aj4.get_abs_index()) == 0) {
                orbit<N, element_type> o4(m_sym, aj4.get_abs_index());
                if(!m_sch->contains(o4.get_acindex())) {
                    m_sch->insert(o4.get_acindex());
                    visit_orbit(o4, visited);
                }
            }

            index<N> j5(aj1.get_index());
            j5.permute(m_perm2).permute(m_perm1);
            abs_index<N> aj5(j5, bidims);
            if(visited.count(aj5.get_abs_index()) == 0) {
                orbit<N, element_type> o5(m_sym, aj5.get_abs_index());
                if(!m_sch->contains(o5.get_acindex())) {
                    m_sch->insert(o5.get_acindex());
                    visit_orbit(o5, visited);
                }
            }

            index<N> j6(aj1.get_index());
            j6.permute(m_perm1).permute(m_perm2).permute(m_perm1);
            abs_index<N> aj6(j6, bidims);
            if(visited.count(aj6.get_abs_index()) == 0) {
                orbit<N, element_type> o6(m_sym, aj6.get_abs_index());
                if(!m_sch->contains(o6.get_acindex())) {
                    m_sch->insert(o6.get_acindex());
                    visit_orbit(o6, visited);
                }
            }
        }
    }

    gen_bto_symmetrize3::stop_timer("make_schedule");
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize3<N, Traits, Timed>::make_schedule_blk(
    const abs_index<N> &ai, sym_schedule_type &sch) const {

    element_type scal = m_symm ? 1.0 : -1.0;
    index<N> idx0(ai.get_index()), idx1(idx0), idx2(idx0), idx3(idx0),
        idx4(idx0), idx5(idx0);
    idx1.permute(m_perm1);
    idx2.permute(m_perm2);
    idx3.permute(m_perm1).permute(m_perm2);
    idx4.permute(m_perm2).permute(m_perm1);
    idx5.permute(m_perm1).permute(m_perm2).permute(m_perm1);

    const symmetry<N, element_type> &sym0 = m_op.get_symmetry();
    const assignment_schedule<N, element_type> &sch0 = m_op.get_schedule();

    orbit<N, double> o0(sym0, idx0), o1(sym0, idx1), o2(sym0, idx2),
        o3(sym0, idx3), o4(sym0, idx4), o5(sym0, idx5);

    //  This is a temporary schedule for the formation of the block
    std::list<schrec> sch1;

    //  Form the temporary schedule

    if(sch0.contains(o0.get_acindex())) {
        tensor_transf<N, element_type> tr(o0.get_transf(idx0));
        sch1.push_back(schrec(o0.get_acindex(), tr));
    }
    if(sch0.contains(o1.get_acindex())) {
        tensor_transf<N, element_type> tr(o1.get_transf(idx1));
        tr.permute(m_perm1);
        tr.transform(scalar_transf<element_type>(scal));
        sch1.push_back(schrec(o1.get_acindex(), tr));
    }
    if(sch0.contains(o2.get_acindex())) {
        tensor_transf<N, element_type> tr(o2.get_transf(idx2));
        tr.permute(m_perm2);
        tr.transform(scalar_transf<element_type>(scal));
        sch1.push_back(schrec(o2.get_acindex(), tr));
    }
    if(sch0.contains(o3.get_acindex())) {
        tensor_transf<N, element_type> tr(o3.get_transf(idx3));
        tr.permute(m_perm2);
        tr.permute(m_perm1);
        sch1.push_back(schrec(o3.get_acindex(), tr));
    }
    if(sch0.contains(o4.get_acindex())) {
        tensor_transf<N, element_type> tr(o4.get_transf(idx4));
        tr.permute(m_perm1);
        tr.permute(m_perm2);
        sch1.push_back(schrec(o4.get_acindex(), tr));
    }
    if(sch0.contains(o5.get_acindex())) {
        tensor_transf<N, element_type> tr(o5.get_transf(idx5));
        tr.permute(m_perm1);
        tr.permute(m_perm2);
        tr.permute(m_perm1);
        tr.transform(scalar_transf<element_type>(scal));
        sch1.push_back(schrec(o5.get_acindex(), tr));
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
            tensor_transf<N, element_type> tr;
            tr.permute(tr0.get_perm());
            tr.transform(scalar_transf<element_type>(c));
            sch.insert(std::make_pair(
                ai.get_abs_index(), schrec(aidx.get_abs_index(), tr)));
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMMETRIZE3_IMPL_H
