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

point_group_table::label_set_t point_group_table::product(label_t l1,
        const label_set_t &l2) const {

    static const char *method = "product(label_t, const label_set_t &) const";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l1))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid label.");

    for (label_set_t::const_iterator it = l2.begin(); it != l2.end(); it++) {
        if (! is_valid(*it))
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid label.");
    }
#endif

    label_set_t ls;
    for (label_set_t::const_iterator it = l2.begin(); it != l2.end(); it++) {
        const label_set_t &lsx = 
            m_table[l1 < *it ? pair_index(l1, *it) : pair_index(*it, l1)];
        ls.insert(lsx.begin(), lsx.end());
    }
    return ls;
}

point_group_table::label_set_t point_group_table::product(const label_set_t &l1,
        label_t l2) const {

    static const char *method = "product(const label_set_t &, label_t) const";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l2))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid label.");

    for (label_set_t::const_iterator it = l1.begin(); it != l1.end(); it++) {
        if (! is_valid(*it))
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid label.");
    }
#endif

    label_set_t ls;
    for (label_set_t::const_iterator it = l1.begin(); it != l1.end(); it++) {
        const label_set_t &lsx = 
            m_table[*it < l2 ? pair_index(*it, l2) : pair_index(l2, *it)];
        ls.insert(lsx.begin(), lsx.end());
    }
    return ls;
}

point_group_table::label_set_t point_group_table::product(const label_set_t &l1,
        const label_set_t &l2) const {

    static const char *method = "product(label_set_t, label_set_t)";

#ifdef LIBTENSOR_DEBUG
    for (label_set_t::const_iterator it = l1.begin(); it != l1.end(); it++) {
        if (! is_valid(*it))
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid label.");
    }
    for (label_set_t::const_iterator it = l2.begin(); it != l2.end(); it++) {
        if (! is_valid(*it))
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid label.");
    }
#endif

    label_set_t ls;
    for (label_set_t::const_iterator it1 = l1.begin();
            it1 != l1.end(); it1++) {
        for (label_set_t::const_iterator it2 = l2.begin();
                it2 != l2.end(); it2++) {
            const label_set_t &lsx = m_table[*it1 < *it2 ? 
                                             pair_index(*it1, *it2) : 
                                             pair_index(*it2, *it1)];
            ls.insert(lsx.begin(), lsx.end());
        }
    }
    return ls;
}

void point_group_table::product(const label_group_t &lg,
        label_set_t &prod) const {

    prod.clear();
    if (lg.empty()) return;

    label_set_t pr1, pr2, *ip1 = &pr1, *ip2 = &pr2;
    label_group_t::const_iterator it = lg.begin();
    ip1->insert(*it);
    it++;
    for (; it != lg.end(); it++) {
        for (label_set_t::const_iterator ii = ip1->begin(); 
             ii != ip1->end(); ii++) {
            const label_set_t &lsx = m_table[*it > *ii ? 
                                             pair_index(*ii, *it) : 
                                             pair_index(*it, *ii)];
            ip2->insert(lsx.begin(), lsx.end());
        }
        std::swap(ip1, ip2);
        ip2->clear();
    }
    prod.swap(*ip1);
}

bool point_group_table::is_in_product(const label_group_t &lg,
        label_t l) const {

    static const char *method =
            "is_in_product(const label_group_t &, label_t)";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l.");
#endif

    if (lg.empty()) return false;

    label_set_t pr1, pr2, *ip1 = &pr1, *ip2 = &pr2;
    label_group_t::const_iterator it = lg.begin();
    ip1->insert(*it);
    it++;
    for (; it != lg.end(); it++) {
        for (label_set_t::const_iterator ii = ip1->begin(); 
             ii != ip1->end(); ii++) {
            const label_set_t &lsx = m_table[*it > *ii ? 
                                             pair_index(*ii, *it) : 
                                             pair_index(*it, *ii)];
            ip2->insert(lsx.begin(), lsx.end());
        }
        std::swap(ip1, ip2);
        ip2->clear();
    }
    return (ip1->count(l) != 0);
}


void point_group_table::initialize_table() {

    for (label_t i = 0; i < m_irreps.size(); i++) {
        label_set_t &ls = m_table[pair_index(k_identity, i)];
        ls.insert(i);
    }
}

} // namespace libtensor


