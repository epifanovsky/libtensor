#ifndef LIBTENSOR_SE_PERM_IMPL_H
#define LIBTENSOR_SE_PERM_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, typename T>
se_perm<N, T>::se_perm(const permutation<N> &perm,
        const scalar_transf<T> &tr) :
        m_transf(perm, tr), m_orderp(1), m_orderc(1) {

    static const char *method =
            "se_perm(const permutation<N>&, const scalar_transf<T>&)";

    if(perm.is_identity() && ! tr.is_identity()) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "perm.is_identity()");
    }

    permutation<N> p(perm);
    while(! p.is_identity()) {
        p.permute(perm);
        m_orderp++;
    }

    scalar_transf<T> trx(tr);
    while (! trx.is_identity() && m_orderc < m_orderp) {
        trx.transform(tr);
        m_orderc++;
    }

    if (! trx.is_identity() || (m_orderp % m_orderc) != 0) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "perm and tr do not agree.");
    }
}

template<size_t N, typename T>
const char *se_perm<N, T>::get_type() const {
    return k_sym_type;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_IMPL_H

