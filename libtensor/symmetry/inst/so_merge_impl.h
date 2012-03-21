#ifndef LIBTENSOR_SO_MERGE_IMPL_H
#define LIBTENSOR_SO_MERGE_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, size_t M, size_t K, typename T>
const char *so_merge<N, M, K, T>::k_clazz = "so_merge<N, M, K, T>";

template<size_t N, size_t M, size_t K, typename T>
void so_merge<N, M, K, T>::add_mask(const mask<N> &msk) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "add_mask(const mask<N> &)";

    size_t nm = 0;
    for (register size_t i = 0; i < N; i++) if (msk[i]) nm++;
    if (nm == 0)
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "msk");

    for (size_t j = 0; j < m_msk_set; j++) {
        register size_t i = 0;
        for (; i < N; i++) if (m_msk[j][i] && msk[i]) break;
        if (i != N)
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "msk");
    }

    if (m_msk_set == K) {
        throw bad_symmetry(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Cannot add any more masks.");
    }

#endif

    m_msk[m_msk_set++] = msk;
}

template<size_t N, size_t M, size_t K, typename T>
void so_merge<N, M, K, T>::perform(symmetry<N - M + K, T> &sym2) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "perform(symmetry<N - M + K, T> &)";
    if (m_msk_set != K) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Masks not set properly.");
    }
#endif

    sym2.clear();

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

        symmetry_element_set<N - M + K, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(set1, m_msk, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N - M + K, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {

            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_H

