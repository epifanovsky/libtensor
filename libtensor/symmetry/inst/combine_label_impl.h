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

    const evaluation_rule<N> &rule = el.get_rule();

    // No products means all forbidden => m_rule becomes empty as well.
    if (rule.begin() == rule.end()) {
        m_rule.clear();
        return;
    }

    // Transfer products
    typename evaluation_rule<N>::const_iterator ij = rule.begin();

    std::list< product_rule<N> > prlist;
    for (typename evaluation_rule<N>::iterator ii = m_rule.begin();
            ii != m_rule.end(); ii++) {

        product_rule<N> &pra = m_rule.get_product(ii);
        prlist.push_back(pra);

        // Amend the product by prb
        const product_rule<N> &prb = rule.get_product(ij);
        for (typename product_rule<N>::iterator ip = prb.begin();
                ip != prb.end(); ip++) {

            pra.add(prb.get_sequence(ip), prb.get_intrinsic(ip));
        }
    }
    ij++;
    for (; ij != rule.end(); ij++) {

        const product_rule<N> &prb = rule.get_product(ij);
        typename std::list< product_rule<N> >::iterator it = prlist.begin();
        for (; it != prlist.end(); it++) {

            product_rule<N> &pra = m_rule.new_product();
            for (typename product_rule<N>::iterator ipa = it->begin();
                    ipa != it->end(); ipa++) {
                pra.add(it->get_sequence(ipa), it->get_intrinsic(ipa));
            }

            for (typename product_rule<N>::iterator ipb = prb.begin();
                    ipb != prb.end(); ipb++) {
                pra.add(prb.get_sequence(ipb), prb.get_intrinsic(ipb));
            }
        }
    }
    m_rule.optimize();
}


} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_IMPL_H
