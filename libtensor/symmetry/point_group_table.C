#include "point_group_table.h"

namespace libtensor {

const char *point_group_table::k_clazz = "point_group_table";

point_group_table::point_group_table(const std::string &id,
        const std::vector<std::string> &irreps, const std::string &identity) :
        m_id(id), m_irreps(irreps.size()),
        m_table(irreps.size() * (irreps.size() + 1) / 2) {

    static const char *method = "point_group_table(const std::string &, "
            "const std::vector<std::string> &, const std::string &)";

    m_irreps[k_identity] = identity;
    label_t l = 1;
    for (size_t i = 0; i < irreps.size(); i++) {
        if (irreps[i] == identity) continue;

        m_irreps[l] = irreps[i];
        l++;
    }
    if (l != irreps.size()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Identity irrep not valid.");
    }

    initialize_table();
}

point_group_table::label_t
point_group_table::get_label(const std::string &irrep) const {

    label_t i = 0;
    for (std::vector<std::string>::const_iterator it = m_irreps.begin();
            it != m_irreps.end(); it++, i++) {
        if (*it == irrep) break;
    }
    if (i == m_irreps.size())
        throw bad_parameter(g_ns, k_clazz,
                "get_label(const std::string &) const",
                __FILE__, __LINE__, "Invalid irrep.");

    return i;
}

void point_group_table::add_product(label_t l1, label_t l2,
        label_t lr) throw(bad_parameter) {

    const char *method = "add_product(label_t, label_t, label_t)";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l1))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l1.");
    if (! is_valid(l2))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l2.");
    if (! is_valid(lr))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep lr.");
#endif

    if (l1 > l2) std::swap(l1, l2);
    if (l1 == product_table_i::k_identity) {
#ifdef LIBTENSOR_DEBUG
        if (l2 != lr)
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "l2 != lr.");
#endif
        return;
    }

    label_set_t &lg = m_table[pair_index(l1, l2)];
    lg.insert(lr);
}

point_group_table::label_set_t
point_group_table::determine_product(label_t l1, label_t l2) const {
#ifdef LIBTENSOR_DEBUG
    if (l1 > l2)
        throw bad_parameter(g_ns, k_clazz,
                "determine_product(label_t, label_t) const",
                __FILE__, __LINE__, "l1 > l2.");
#endif

    return m_table[pair_index(l1, l2)];
}


void point_group_table::do_check() const {

}

void point_group_table::initialize_table() {

    for (label_t i = 0; i < m_irreps.size(); i++) {
        label_set_t &ls = m_table[pair_index(k_identity, i)];
        ls.insert(i);
    }
}

} // namespace libtensor


