#ifndef LIBTENSOR_COMBINE_LABEL_IMPL_H
#define LIBTENSOR_COMBINE_LABEL_IMPL_H

#include "er_optimize.h"

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

    const evaluation_rule<N> &r1 = el.get_rule();

    // No products means all forbidden => m_rule becomes empty as well.
    if (r1.begin() == r1.end()) {
        m_rule.clear();
        return;
    }

    // Copy current rule and clear it
    evaluation_rule<N> r2;

    // Transfer products
    for (typename evaluation_rule<N>::iterator ia = m_rule.begin();
            ia != m_rule.end(); ia++) {

        const product_rule<N> &pra = m_rule.get_product(ia);

        for (typename evaluation_rule<N>::iterator ib = r1.begin();
                ib != r1.end(); ib++) {

            const product_rule<N> &prb = r1.get_product(ib);

            product_rule<N> &prc = r2.new_product();
            for (typename product_rule<N>::iterator ipa = pra.begin();
                    ipa != pra.end(); ipa++) {

                prc.add(pra.get_sequence(ipa), pra.get_intrinsic(ipa));
            }
            for (typename product_rule<N>::iterator ipb = prb.begin();
                    ipb != prb.end(); ipb++) {

                prc.add(prb.get_sequence(ipb), prb.get_intrinsic(ipb));
            }
        }
    }
    m_rule.clear();
    er_optimize<N>(r2, m_table_id).perform(m_rule);
}


} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_IMPL_H
