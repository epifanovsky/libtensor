#ifndef LIBTENSOR_SO_REDUCE_IMPL_H
#define LIBTENSOR_SO_REDUCE_IMPL_H

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *so_reduce<N, M, T>::k_clazz = "so_reduce<N, M, K, T>";

template<size_t N, size_t M, typename T>
so_reduce<N, M, T>::so_reduce(const symmetry<N, T> &sym1, const mask<N> &msk,
        const sequence<N, size_t> &rseq, const index_range<N> &rrange) :
    m_sym1(sym1), m_msk(msk), m_rseq(rseq), m_rrange(rrange) {

    static const char *method = "so_reduce(const symmetry<N, T> &, "
            "const mask<N> &, const sequence<N, size_t> &, "
            "const index_range<N> &)";

#ifdef LIBTENSOR_DEBUG
    // Check the mask, the sequence, and the index range
    size_t m = 0;
    mask<N> rsets;
    index<N> ria, rib;
    register size_t i = 0;
    for (; i < N; i++) {
        if (! msk[i]) continue;
        if (rseq[i] > N) break;

        rsets[rseq[i]] = true;
        ria[rseq[i]] = rrange.get_begin()[i];
        rib[rseq[i]] = rrange.get_end()[i];
        m++;
    }
    if (i != N) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "rseq.");
    }
    if (m != M) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "msk.");
    }

    i = 0;
    for (; i < N && rsets[i]; i++) {
        register size_t j = 0;
        for (; j < N; j++) {
            if (! msk[j] || rseq[j] != i) continue;
            if (rrange.get_begin()[j] != ria[i]) break;
            if (rrange.get_end()[j] != rib[i]) break;
        }
        if (j != N) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "rrange.");
        }
    }
    for (; i < N && ! rsets[i]; i++) { }
    if (i != N) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "rseq.");
    }

#endif
}


template<size_t N, size_t M, typename T>
void so_reduce<N, M, T>::perform(symmetry<N - M, T> &sym2) {

    sym2.clear();

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 =
                m_sym1.get_subset(i);
        symmetry_element_set<N - M, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(
                set1, m_msk, m_rseq, m_rrange, set2);

        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N - M, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_H

