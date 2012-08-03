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

#ifdef LIBTENSOR_DEBUG
    if (el.get_table_id() != m_table_id) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Table ID.");
    }

    if (! (el.get_labeling() == m_blk_labels)) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Block labels.");
    }
#endif

    typedef typename evaluation_rule<N>::iterator product_iterator;
    typedef typename evaluation_rule<N>::const_iterator product_const_iterator;
    typedef typename evaluation_rule<N>::product_rule_t product_rule_t;

    const evaluation_rule<N> &rule = el.get_rule();

    // No products means all forbidden => m_rule becomes empty as well.
    if (rule.begin() == rule.end()) {
        m_rule.clear(); return;
    }

    // Transfer products
    for (product_iterator ii = m_rule.begin(); ii != m_rule.end(); ii++) {

        // Current product
        product_rule_t &pra = m_rule.get_product(ii);
        product_rule_t pra_copy(pra);

        product_const_iterator ij = rule.begin();
        {
            const product_rule_t &prb = rule.get_product(ij);
            // Amend the product by prb
            for (typename product_rule_t::iterator ip = prb.begin();
                    ip != prb.end(); ip++) {

                pra.add(prb.get_sequence(ip), prb.get_intrinsic(ip));
            }
            ij++;
        }

        for (; ij != rule.end(); ij++) {

            product_rule_t &prx = m_rule.new_product();

            for (typename product_rule_t::iterator ip = pra_copy.begin();
                    ip != pra_copy.end(); ip++) {

                prx.add(pra_copy.get_sequence(ip), pra_copy.get_intrinsic(ip));
            }

            const product_rule_t &prb = rule.get_product(ij);
            for (typename product_rule_t::iterator ip = prb.begin();
                    ip != ij->end(); ip++) {

                prx.add(prb.get_sequence(ip), prb.get_intrinsic(ip));
            }
        }

    }
}


} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_IMPL_H
