#include "point_group_table.h"

namespace libtensor {

const char *point_group_table::k_clazz = "point_group_table";

point_group_table::point_group_table(const std::string &id,
        const irrep_map_t &irrep_names, irrep_label_t ident) :
        m_id(id), m_irrep_names(irrep_names), m_id_irrep(ident)
{
    static const char *method = "point_group_table(const std::string &, "
            "const irrep_map_t &, label_t)";

    if (m_irrep_names.find(m_id_irrep) == m_irrep_names.end())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Identity irrep not valid.");

    for (irrep_map_t::const_iterator it = m_irrep_names.begin();
            it != m_irrep_names.end(); it++) {

        m_irreps.insert(it->first);
    }

    init_table();
}

point_group_table::point_group_table(const point_group_table &pt) :
        m_id(pt.m_id), m_irrep_names(pt.m_irrep_names), m_irreps(pt.m_irreps),
        m_id_irrep(pt.m_id_irrep), m_table(pt.m_table) { }

bool point_group_table::is_in_product(
        const irrep_group_t &lg, irrep_label_t l) const {

    static const char *method =
            "is_in_product(const irrep_group_t &, irrep_label_t)";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l))
        throw out_of_bounds(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l.");
#endif

    if (lg.empty()) return false;

    irrep_group_t::const_iterator it = lg.begin();
    irrep_set_t ls; ls.insert(*it);
    it++;

    for (; it != lg.end(); it++) ls = product(*it, ls);

    return (ls.find(l) != ls.end());
}

point_group_table::irrep_set_t point_group_table::product(
        const irrep_set_t &ls1, const irrep_set_t &ls2) const {

    irrep_set_t ls;
    for (irrep_set_t::const_iterator it1 = ls1.begin();
            it1 != ls1.end(); it1++) {

        for (irrep_set_t::const_iterator it2 = ls2.begin();
                it2 != ls2.end(); it2++) {

            table_t::const_iterator itx = m_table.find(pair_index(*it1, *it2));
#ifdef LIBTENSOR_DEBUG
            if (itx == m_table.end()) {
                throw generic_exception(g_ns, k_clazz,
                        "product(const irrep_set_t &, const irrep_set_t &)",
                        __FILE__, __LINE__, "Incomplete table.");
            }
#endif

            ls.insert(itx->second.begin(), itx->second.end());
        }
    }
    return ls;
}

void point_group_table::add_product(irrep_label_t l1, irrep_label_t l2,
        irrep_label_t lr) throw(bad_parameter) {

    const char *method =
            "add_product(irrep_label_t, irrep_label_t, irrep_label_t)";

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

    irrep_set_t &lg = m_table[pair_index(l1, l2)];
    lg.insert(lr);
}

void point_group_table::check() const throw(generic_exception) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "check()";

    for (irrep_set_t::const_iterator it1 = m_irreps.begin();
            it1 != m_irreps.end(); it1++) {

        for (irrep_set_t::const_iterator it2 = it1;
                it2 != m_irreps.end(); it2++) {

            table_t::const_iterator itx =
                    m_table.find(pair_index(*it1, *it2));

            if (itx == m_table.end())
                generic_exception(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Product table not setup.");

            for (irrep_set_t::const_iterator it3 = itx->second.begin();
                    it3 != itx->second.end(); it3++) {
                if (! is_valid(*it3))
                    generic_exception(g_ns, k_clazz, method,
                            __FILE__, __LINE__, "Invalid irrep.");
            }
        }
    }

    product_table_i::check();
#endif
}

void point_group_table::init_table() {

    for (irrep_set_t::const_iterator it = m_irreps.begin();
            it != m_irreps.end(); it++) {

        irrep_set_t &ls1 = m_table[pair_index(*it, *it)];
        ls1.insert(m_id_irrep);

        if (*it == m_id_irrep) continue;

        irrep_set_t &ls2 = m_table[pair_index(m_id_irrep, *it)];
        ls2.insert(*it);
    }
}

} // namespace libtensor


