#ifndef LIBTENSOR_EVALUATION_RULE_IMPL_H
#define LIBTENSOR_EVALUATION_RULE_IMPL_H

namespace libtensor {

template<size_t N>
const char *evaluation_rule<N>::k_clazz = "evaluation_rule<N>";

template<size_t N>
size_t evaluation_rule<N>::add_sequence(const sequence<N, size_t> &seq) {

    size_t seqno = 0;
    for (; seqno < m_sequences.size(); seqno++) {
        const sequence<N, size_t> &ref = m_sequences[seqno];

        size_t i = 0;
        for (; i < N; i++) {
            if (seq[i] != ref[i]) break;
        }
        if (i == N) return seqno;
    }

    m_sequences.push_back(seq);
    return m_sequences.size() - 1;
}

template<size_t N>
size_t evaluation_rule<N>::add_product(size_t seq_no, label_t target) {
#ifdef LIBTENSOR_DEBUG
    if (seq_no >= m_sequences.size())
        throw bad_parameter(g_ns, k_clazz, "add_product(size_t, label_t)",
                __FILE__, __LINE__, "seq_no.");
#endif

    m_setup.push_back(product_t());
    product_t &pr = m_setup.back();
    pr.insert(product_t::value_type(seq_no, target));
    return m_setup.size() - 1;
}

template<size_t N>
void evaluation_rule<N>::add_to_product(size_t no,
        size_t seq_no, label_t target) {

    static const char *method = "add_to_product(size_t, size_t, label_t)";

#ifdef LIBTENSOR_DEBUG
    if (no >= m_setup.size())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "no");
    if (seq_no >= m_sequences.size())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "seq_no");
#endif

    product_t &pr = m_setup[no];
    pr.insert(product_t::value_type(seq_no, target));
}

template<size_t N>
bool evaluation_rule<N>::is_valid(iterator it) const {

    for (std::vector<product_t>::const_iterator it1 = m_setup.begin();
            it1 != m_setup.end(); it1++) {

        const product_t &pr = *it1;
        for (iterator it2 = pr.begin(); it2 != pr.end(); it2++) {

            if (it == it2) return true;
        }
    }

    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
