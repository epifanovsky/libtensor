#ifndef LIBTENSOR_SO_MERGE_IMPL_H
#define LIBTENSOR_SO_MERGE_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *so_merge<N, M, T>::k_clazz = "so_merge<N, M, T>";

template<size_t N, size_t M, typename T>
so_merge<N, M, T>::so_merge(const symmetry<N, T> &sym1,
        const mask<N> &msk, const sequence<N, size_t> &mseq) :
    m_sym1(sym1), m_msk(msk), m_mseq(mseq) {

    static const char *method = "so_merge(const symmetry<N, T> &, "
            "const mask<N> &, const sequence<N, size_t> &)";

#ifdef LIBTENSOR_DEBUG
    // Check the mask, and the sequence
    size_t m = 0;
    mask<M> msets;
    size_t i = 0;
    for (; i < N; i++) {
        if (! msk[i]) continue;
        if (mseq[i] > M) break;

        msets[mseq[i]] = true;
        m++;
    }
    if (i != N) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "mseq.");
    }
    for (i = 0; i < M && msets[i]; i++) {
        m--;
    }
    for (; i < M && ! msets[i]; i++) { }
    if (i != M) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "mseq.");
    }
    if (m != M) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "msk.");
    }
#endif
}

template<size_t N, size_t M, typename T>
void so_merge<N, M, T>::perform(symmetry<N - M, T> &sym2) {

    sym2.clear();
    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

        symmetry_element_set<N - M, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(set1,
                m_msk, m_mseq, set2);

        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N - M, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_H

