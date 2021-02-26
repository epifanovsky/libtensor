#ifndef LIBTENSOR_SO_REDUCE_IMPL_H
#define LIBTENSOR_SO_REDUCE_IMPL_H

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *so_reduce<N, M, T>::k_clazz = "so_reduce<N, M, K, T>";

template<size_t N, size_t M, typename T>
so_reduce<N, M, T>::so_reduce(const symmetry<N, T> &sym1,
        const mask<N> &msk, const sequence<N, size_t> &rseq,
        const index_range<N> &rblrange, const index_range<N> &riblrange) :
    m_sym1(sym1), m_msk(msk), m_rseq(rseq),
    m_rblrange(rblrange), m_riblrange(riblrange) {

    static const char *method = "so_reduce(const symmetry<N, T> &, "
            "const mask<N> &, const sequence<N, size_t> &, "
            "const index_range<N> &, const index_range<N> &)";

#ifdef LIBTENSOR_DEBUG
    // Check the mask, the sequence, and the index range
    size_t m = 0;
    mask<M> rsets;
    index<M> rbia, rbib, ria, rib;
    size_t i = 0;
    for (; i < N; i++) {
        if (! msk[i]) continue;
        if (rseq[i] > M) break;

        rsets[rseq[i]] = true;
        rbia[rseq[i]] = rblrange.get_begin()[i];
        rbib[rseq[i]] = rblrange.get_end()[i];
        ria[rseq[i]] = riblrange.get_begin()[i];
        rib[rseq[i]] = riblrange.get_end()[i];
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

    for (i = 0; i < M && rsets[i]; i++) {
        size_t j = 0;
        for (; j < N; j++) {
            if (! msk[j] || rseq[j] != i) continue;
            if (rblrange.get_begin()[j] != rbia[i] ||
                    rblrange.get_end()[j] != rbib[i] ||
                    riblrange.get_begin()[j] != ria[i] ||
                    riblrange.get_end()[j] != rib[i]) break;
        }
        if (j != N) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "rrange.");
        }
    }
    for (; i < M && ! rsets[i]; i++) { }
    if (i != M) {
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
                set1, m_msk, m_rseq, m_rblrange, m_riblrange, set2);

        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N - M, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_H

