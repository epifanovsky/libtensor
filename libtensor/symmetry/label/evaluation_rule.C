#include "evaluation_rule.h"

namespace libtensor {

const char *evaluation_rule::k_clazz = "evaluation_rule";

const size_t evaluation_rule::k_intrinsic = (size_t) -1;

evaluation_rule::rule_id evaluation_rule::add_rule(
        const label_set &intr, const std::vector<size_t> &order) {

    rule_id id = new_rule_id();
    m_rules.insert(rule_list::value_type(id, basic_rule(intr, order)));
    return id;
}

size_t evaluation_rule::add_product(rule_id rule) {

    rule_iterator it = m_rules.find(rule);
    if (it == m_rules.end())
        throw bad_parameter(g_ns, k_clazz,
                "add_product(rule_id)", __FILE__, __LINE__, "rule");

    m_setup.push_back(rules_product());
    rules_product &pr = m_setup.back();
    pr[rule] = it;
    return m_setup.size() - 1;
}

void evaluation_rule::add_to_product(size_t no, rule_id rule) {

    static const char *method = "add_to_product(size_t, rule_id)";

#ifdef LIBTENSOR_DEBUG
    if (no >= m_setup.size())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "no");
#endif

    rule_iterator it = m_rules.find(rule);
    if (it == m_rules.end())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "rule");

    rules_product &pr = m_setup[no];
    pr[rule] = it;
}

bool evaluation_rule::is_valid_rule(rule_iterator it) const {

    for (rule_iterator i = m_rules.begin(); i != m_rules.end(); i++) {

        if (it == i) return true;
    }

    return false;
}

bool evaluation_rule::is_valid_product_iterator(product_iterator it) const {

    for (product_list::const_iterator i = m_setup.begin();
            i != m_setup.end(); i++) {

        const rules_product &p = *i;
        for (product_iterator j = p.begin(); j != p.end(); j++) {

            if (it == j) return true;
        }
    }

    return false;
}


} // namespace libtensor


