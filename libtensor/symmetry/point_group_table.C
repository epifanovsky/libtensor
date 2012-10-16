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

    m_table[pair_index(l1, l2)] |= (1 << lr);
}


void point_group_table::product(const label_group_t &lg,
        label_set_t &prod) const {

    prod.clear();
    if (lg.empty()) return;

    label_group_t::const_iterator it = lg.begin();
    register size_t pr1 = (1 << *it);
    it++;
    for (; it != lg.end(); it++) {
        register size_t pr2 = 0;
        for (register label_t i = 0, bit = 1;
                i < m_irreps.size(); i++, bit <<= 1) {
            if ((pr1 & bit) == bit)
                pr2 |= m_table[*it > i ?
                        pair_index(i, *it) : pair_index(*it, i)];
        }
        pr1 = pr2;
    }

    for (register label_t i = 0, bit = 1; i < m_irreps.size(); i++, bit <<= 1) {
        if ((pr1 & bit) == bit) prod.insert(i);
    }
}

bool point_group_table::is_in_product(
        const label_group_t &lg, label_t l) const {

    static const char *method =
            "is_in_product(const label_group_t &, label_t)";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l.");
#endif

    if (lg.empty()) return false;

    label_group_t::const_iterator it = lg.begin();
    register size_t pr1 = (1 << *it);
    it++;
    for (; it != lg.end(); it++) {
        register size_t pr2 = 0;
        for (register size_t i = 0, bit = 1;
                i < m_irreps.size(); i++, bit <<= 1) {
            if ((pr1 & bit) == bit)
                pr2 |= m_table[*it > i ?
                        pair_index(i, *it) : pair_index(*it, i)];
        }
        pr1 = pr2;
    }

    register size_t rbit = (1 << l);
    return (pr1 & rbit) == rbit;
}


void point_group_table::initialize_table() {

    for (label_t i = 0; i < m_irreps.size(); i++) {
        m_table[pair_index(k_identity, i)] = (1 << i);
    }
}

} // namespace libtensor


