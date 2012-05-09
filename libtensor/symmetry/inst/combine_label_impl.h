#ifndef LIBTENSOR_COMBINE_LABEL_IMPL_H
#define LIBTENSOR_COMBINE_LABEL_IMPL_H

namespace libtensor {

template<size_t N, typename T>
const char *combine_label<N, T>::k_clazz = "combine_label<N, T>";

template<size_t N, typename T>
combine_label<N, T>::combine_label(const se_label<N, T> &el) :
    m_table_id(el.get_table_id()), m_blk_labels(el.get_labeling()),
    m_rule(el.get_rule()) {

}

template<size_t N, typename T>
void combine_label<N, T>::add(const se_label<N, T> &el) throw(bad_parameter) {

    static const char *method = "add(const se_label<N, T> &)";

    if (el.get_table_id() != m_table_id) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Table ID.");
    }

    if (! (el.get_labeling() == m_blk_labels)) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Block labels.");
    }

    const evaluation_rule<N> &rule = el.get_rule();

    // No products means all forbidden => m_rule becomes empty as well.
    if (rule.get_n_products() == 0) m_rule.clear_all();

    // Transfer sequences
    std::vector<size_t> seqno(rule.get_n_sequences());
    for (size_t i = 0; i < rule.get_n_sequences(); i++) {
        seqno[i] = m_rule.add_sequence(rule[i]);
    }

    // Transfer products
    for (size_t i = 0; i < m_rule.get_n_products(); i++) {

        size_t j = 0;
        for (typename evaluation_rule<N>::iterator it = rule.begin(j);
                it != rule.end(j); it++) {
            m_rule.add_to_product(i, seqno[rule.get_seq_no(it)],
                    rule.get_intrinsic(it), rule.get_target(it));
        }
        j++;

        for (; j < m_rule.get_n_products(); j++) {
            typename evaluation_rule<N>::iterator it = m_rule.begin(i);
            size_t pno = m_rule.add_product(m_rule.get_seq_no(it),
                    rule.get_intrinsic(it), rule.get_target(it));
            it++;
            for (; it != m_rule.end(i); it++) {
                m_rule.add_to_product(pno, m_rule.get_seq_no(it),
                        m_rule.get_intrinsic(it), m_rule.get_target(it));
            }
            for (it = rule.begin(j); it != rule.end(j); j++) {
                m_rule.add_to_product(pno, seqno[rule.get_seq_no(it)],
                        rule.get_intrinsic(it), rule.get_target(it));
            }
        }
    }

    m_rule.optimize();
}

} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_IMPL_H
