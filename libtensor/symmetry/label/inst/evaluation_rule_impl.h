#ifndef LIBTENSOR_EVALUATION_RULE_IMPL_H
#define LIBTENSOR_EVALUATION_RULE_IMPL_H

namespace libtensor {

template<size_t N>
const char *evaluation_rule<N>::k_clazz = "evaluation_rule<N>";

template<size_t N>
typename evaluation_rule<N>::rule_id_t
evaluation_rule<N>::add_rule(const basic_rule_t &br) {

    rule_id_t id = new_rule_id();
    m_rules.insert(typename rule_list_t::value_type(id, br));
    return id;
}

template<size_t N>
size_t evaluation_rule<N>::add_product(rule_id_t rule) {

    rule_iterator it = m_rules.find(rule);
    if (it == m_rules.end())
        throw bad_parameter(g_ns, k_clazz,
                "add_product(rule_id)", __FILE__, __LINE__, "rule");

    m_setup.push_back(rule_product_t());
    rule_product_t &pr = m_setup.back();
    pr[rule] = it;
    return m_setup.size() - 1;
}

template<size_t N>
void evaluation_rule<N>::add_to_product(size_t no, rule_id_t rule) {

    static const char *method = "add_to_product(size_t, rule_id)";

#ifdef LIBTENSOR_DEBUG
    if (no >= m_setup.size())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "no");
#endif

    rule_iterator it = m_rules.find(rule);
    if (it == m_rules.end())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "rule");

    rule_product_t &pr = m_setup[no];
    pr[rule] = it;
}

template<size_t N>
bool evaluation_rule<N>::is_valid_rule(rule_iterator it) const {

    for (rule_iterator i = m_rules.begin(); i != m_rules.end(); i++) {

        if (it == i) return true;
    }

    return false;
}

template<size_t N>
bool evaluation_rule<N>::is_valid_product_iterator(product_iterator it) const {

    for (typename product_list_t::const_iterator i = m_setup.begin();
            i != m_setup.end(); i++) {

        const rule_product_t &p = *i;
        for (product_iterator j = p.begin(); j != p.end(); j++) {

            if (it == j) return true;
        }
    }

    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
