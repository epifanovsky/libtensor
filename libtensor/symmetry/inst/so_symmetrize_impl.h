#ifndef LIBTENSOR_SO_SYMMETRIZE_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_IMPL_H

#include "../so_copy.h"

namespace libtensor {

template<size_t N, typename T>
const char *so_symmetrize<N, T>::k_clazz = "so_symmetrize<N, T>";

template<size_t N, typename T>
so_symmetrize<N, T>::so_symmetrize(const symmetry<N, T> &sym1,
        const sequence<N, size_t> &idxgrp, const sequence<N, size_t> &symidx,
        const scalar_transf<T> &pt, const scalar_transf<T> &ct) :
        m_sym1(sym1), m_idxgrp(idxgrp), m_symidx(symidx),
        m_trp(pt), m_trc(ct) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "so_symmetrize(const symmetry<N, T> &, "
            "const sequence<N, size_t> &, const sequence<N, size_t> &, bool)";

    size_t ngrp = 0, nidx = 0;
    sequence<N, size_t> nidxs(0), ngrps(0);
    for (register size_t i = 0; i < N; i++) {
        if (m_idxgrp[i] == 0 || m_symidx[i] == 0) {
            if (m_idxgrp[i] != m_symidx[i]) {
                throw bad_symmetry(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Inconsistent sequences.");
            }
            continue;
        }

        if (m_idxgrp[i] > N) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "idxgrp.");
        }
        if (m_symidx[i] > N) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "symidx.");
        }

        ngrp = std::max(ngrp, m_idxgrp[i]);
        nidx = std::max(nidx, m_symidx[i]);

        nidxs[m_idxgrp[i] - 1]++;
        ngrps[m_symidx[i] - 1]++;
    }

    register size_t j = 0;
    for (; j < N && (ngrps[j] != 0); j++) {
        if (ngrps[j] != ngrp) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "# groups in symidx.");
        }
    }
    for (; j < N && (ngrps[j] == 0); j++) ;
    if (j < N) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "symidx.");
    }

    for (j = 0; j < N && (nidxs[j] != 0); j++) {
        if (nidxs[j] != nidx) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "# indexes in idxgrp.");
        }
    }
    for (; j < N && (nidxs[j] == 0); j++) ;
    if (j < N) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "idxgrp.");
    }

    scalar_transf<T> trp(pt), trc(ct);
    trp.transform(pt);
    if (! trp.is_identity())
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "pt.");

    for (j = 1; j < ngrp; j++) trc.transform(ct);
    if (! trc.is_identity())
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "ct.");
#endif
}

template<size_t N, typename T>
void so_symmetrize<N, T>::perform(symmetry<N, T> &sym2) {

    size_t ngrp = 0;
    for (register size_t i = 0; i < N; i++) {
        if (m_idxgrp[i] == 0) continue;
        ngrp = std::max(ngrp, m_idxgrp[i]);
    }
    if (ngrp < 2) {
        so_copy<N, T>(m_sym1).perform(sym2);
        return;
    }

    sym2.clear();

    bool perm_done = false;
    for(typename symmetry<N, T>::iterator i1 = m_sym1.begin();
            i1 != m_sym1.end(); i1++) {

        const symmetry_element_set<N, T> &set1 =
                m_sym1.get_subset(i1);
        if(set1.get_id().compare(se_perm<N, T>::k_sym_type) == 0) {
            perm_done = true;
        }

        symmetry_element_set<N, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(set1,
                m_idxgrp, m_symidx, m_trp, m_trc, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }

    //  If the argument does not have any permutation symmetry
    //  elements, the handler does not get called above. We need to
    //  fix that manually.
    if(! perm_done) {
        symmetry_element_set<N, T> set1(se_perm<N, T>::k_sym_type),
                set2(se_perm<N, T>::k_sym_type);
        symmetry_operation_params<operation_t> params(set1,
                m_idxgrp, m_symidx, m_trp, m_trc, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_IMPL_H
