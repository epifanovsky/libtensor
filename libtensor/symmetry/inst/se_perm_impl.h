#ifndef LIBTENSOR_SE_PERM_IMPL_H
#define LIBTENSOR_SE_PERM_IMPL_H

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
    if(!m_even && !m_symm)
        m_transf.transform(scalar_transf<T>(-1.0));
}


template<size_t N, typename T>
se_perm<N, T>::se_perm(const se_perm<N, T> &elem) :
    m_perm(elem.m_perm), m_symm(elem.m_symm), m_transf(elem.m_transf) {

}
//
//template<size_t N, typename T, typename Tr>
//const char *se_perm<N, T, Tr>::k_clazz = "se_perm<N, T, Tr>";
//
//template<size_t N, typename T, typename Tr>
//const char *se_perm<N, T, Tr>::k_sym_type = "perm";
//
//template<size_t N, typename T, typename Tr>
//se_perm<N, T, Tr>::se_perm(const permutation<N> &perm, unsigned char n) :
//        m_perm(perm), m_n(n % Tr::k_order) {
//
//    static const char *method =
//            "se_perm(const permutation<N>&, unsigned char)";
//
//    if(perm.is_identity() && m_n != 0) {
//        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
//                "perm.is_identity()");
//    }
//
//    unsigned char nn = m_n;
//    permutation<N> p(m_perm);
//    do {
//        p.permute(m_perm);
//        nn = (nn + m_n) % Tr::k_order;
//    } while(!p.is_identity());
//
//    if (nn != 0) {
//        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
//                "perm and n do not agree.");
//    }
//
//    m_transf.permute(m_perm);
//    for (size_t i = 0; i < m_n; i++) {
//        m_transf.transform(Tr::k_generator);
//    }
//}
//
//
//template<size_t N, typename T, typename Tr>
//se_perm<N, T, Tr>::se_perm(const se_perm<N, T> &elem) :
//    m_perm(elem.m_perm), m_n(elem.m_n), m_transf(elem.m_transf) {
//
//}

} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_IMPL_H

