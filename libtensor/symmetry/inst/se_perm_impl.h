#ifndef LIBTENSOR_SE_PERM_IMPL_H
#define LIBTENSOR_SE_PERM_IMPL_H

#include <libtensor/defs.h>
#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, typename T>
const char *se_perm<N, T>::k_clazz = "se_perm<N, T>";

template<size_t N, typename T>
const char *se_perm<N, T>::k_sym_type = "perm";

template<size_t N, typename T>
se_perm<N, T>::se_perm(const permutation<N> &perm, bool symm) :
m_perm(perm), m_symm(symm) {

    static const char *method = "se_perm(const permutation<N>&, bool)";

    if(perm.is_identity()) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "perm.is_identity()");
    }

    size_t n = 0;
    permutation<N> p(m_perm);
    do {
        p.permute(m_perm); n++;
    } while(!p.is_identity());

    m_even = n % 2 == 0;
    m_transf.permute(m_perm);
    if(!m_even && !m_symm) m_transf.scale(-1);
}


template<size_t N, typename T>
se_perm<N, T>::se_perm(const se_perm<N, T> &elem) :
m_perm(elem.m_perm), m_symm(elem.m_symm), m_transf(elem.m_transf) {

}

template<size_t N, typename T>
inline bool se_perm<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    block_index_space<N> bis2(bis);
    bis2.permute(m_perm);
    return bis2.equals(bis);
}

template<size_t N, typename T>
inline void se_perm<N, T>::apply(index<N> &idx) const {

    idx.permute(m_transf.get_perm());
}

template<size_t N, typename T>
inline void se_perm<N, T>::apply(index<N> &idx, transf<N, T> &tr) const {

    idx.permute(m_transf.get_perm());
    tr.transform(m_transf);
}


} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_IMPL_H
