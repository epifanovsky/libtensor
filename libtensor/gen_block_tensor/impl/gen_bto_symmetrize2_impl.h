#ifndef LIBTENSOR_GEN_BTO_SYMMETRIZE2_IMPL_H
#define LIBTENSOR_GEN_BTO_SYMMETRIZE2_IMPL_H

#include <list>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include "../gen_bto_aux_symmetrize.h"
#include "../gen_bto_symmetrize2.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_symmetrize2<N, Traits, Timed>::k_clazz[] =
    "gen_bto_symmetrize2<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_symmetrize2<N, Traits, Timed>::gen_bto_symmetrize2(
    additive_gen_bto<N, bti_traits> &op,
    const permutation<N> &perm,
    bool symm) :

    m_op(op), m_symm(symm), m_perm1(perm), m_bis(op.get_bis()),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    static const char *method =
        "gen_bto_symmetrize2(additive_bto<N, btod_traits>&, "
        "const permutation<N>&, bool)";

    permutation<N> p1(perm); p1.permute(perm);
    if(perm.is_identity() || !p1.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm");
    }
    make_symmetry();
    make_schedule();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize2<N, Traits, Timed>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    gen_bto_symmetrize2::start_timer();

    try {

        tensor_transf<N, element_type> tr0;
        tensor_transf<N, element_type> tr1(m_perm1,
            scalar_transf<element_type>(m_symm ? 1.0 : -1.0));
        gen_bto_aux_symmetrize<N, Traits> out2(m_op.get_symmetry(), m_sym, out);
        out2.add_transf(tr0);
        out2.add_transf(tr1);
        out2.open();
        m_op.perform(out2);
        out2.close();

    } catch(...) {
        gen_bto_symmetrize2::stop_timer();
        throw;
    }

    gen_bto_symmetrize2::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize2<N, Traits, Timed>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    typedef typename Traits::template temp_block_type<N>::type temp_block_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy;

    typedef typename sym_schedule_type::iterator iterator_t;

    bool zero1 = zero;

    dimensions<N> bidims(m_bis.get_block_index_dims());
    abs_index<N> aib(ib, bidims);

    std::list<schrec> sch1;
    std::pair<iterator_t, iterator_t> jr =
        m_sym_sch.equal_range(aib.get_abs_index());
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

        tensor_transf<N, element_type> tri(sch1.front().tr);
        tri.transform(trb);

        if(n == 1) {

            m_op.compute_block(zero1, ai.get_index(), tri, blkb);
            zero1 = false;
            sch1.pop_front();
        } else {

            temp_block_type tblk(blkb.get_dims());
            m_op.compute_block(true, ai.get_index(), tri, tblk);
            tensor_transf<N, element_type> tri_inv(tri);
            tri_inv.invert();
            for(typename std::list<schrec>::iterator j = sch1.begin();
                j != sch1.end();) {
                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, element_type> trj(tri_inv);
                trj.transform(j->tr);
                trj.transform(trb);
                to_copy(tblk, trj).perform(zero1, blkb);
                zero1 = false;
                j = sch1.erase(j);
            }
        }
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize2<N, Traits, Timed>::make_symmetry() {

    sequence<N, size_t> seq2(0), idxgrp(0), symidx(0);
    for(register size_t i = 0; i < N; i++) seq2[i] = i;
    m_perm1.apply(seq2);

    size_t idx = 1;
    for(register size_t i = 0; i < N; i++) {
        if(seq2[i] <= i) continue;
        idxgrp[i] = 1;
        idxgrp[seq2[i]] = 2;
        symidx[i] = symidx[seq2[i]] = idx++;
    }
    scalar_transf<element_type> tr(m_symm ? 1.0 : -1.0);
    so_symmetrize<N, element_type>(m_op.get_symmetry(), idxgrp, symidx, tr, tr).
        perform(m_sym);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize2<N, Traits, Timed>::make_schedule() {

    gen_bto_symmetrize2::start_timer("make_schedule");

    dimensions<N> bidims(m_bis.get_block_index_dims());

    const assignment_schedule<N, element_type> &sch0 = m_op.get_schedule();
    for(typename assignment_schedule<N, element_type>::iterator i =
        sch0.begin(); i != sch0.end(); ++i) {

        abs_index<N> ai0(sch0.get_abs_index(i), bidims);
        orbit<N, element_type> o(m_op.get_symmetry(), ai0.get_index());

        for(typename orbit<N, element_type>::iterator j = o.begin();
            j != o.end(); j++) {

            abs_index<N> aj1(o.get_abs_index(j), bidims);
            index<N> j2(aj1.get_index()); j2.permute(m_perm1);
            abs_index<N> aj2(j2, bidims);
            short_orbit<N, element_type> so1(m_sym, aj1.get_abs_index());
            short_orbit<N, element_type> so2(m_sym, aj2.get_abs_index());

            if(so1.get_acindex() == aj1.get_abs_index()) {
                if(!m_sch.contains(aj1.get_abs_index())) {
                    m_sch.insert(aj1.get_abs_index());
                }
                tensor_transf<N, element_type> tr1(o.get_transf(j));
                m_sym_sch.insert(std::make_pair(
                    aj1.get_abs_index(), schrec(ai0.get_abs_index(), tr1)));
            }
            if(so2.get_acindex() == aj2.get_abs_index()) {
                if(!m_sch.contains(aj2.get_abs_index())) {
                    m_sch.insert(aj2.get_abs_index());
                }
                tensor_transf<N, element_type> tr2(o.get_transf(j));
                tr2.permute(m_perm1);
                tr2.transform(scalar_transf<element_type>(m_symm ? 1.0 : -1.0));
                m_sym_sch.insert(std::make_pair(
                    aj2.get_abs_index(), schrec(ai0.get_abs_index(), tr2)));
            }
        }
    }

    gen_bto_symmetrize2::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMMETRIZE2_IMPL_H
