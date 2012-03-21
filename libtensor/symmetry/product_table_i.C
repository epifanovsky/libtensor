#include "product_table_i.h"

namespace libtensor {

const char *product_table_i::k_clazz = "product_table_i";

const product_table_i::label_t product_table_i::k_invalid =
        (product_table_i::label_t) -1;

const product_table_i::label_t product_table_i::k_identity = 0;

product_table_i::label_set_t product_table_i::product(label_t l1,
        label_t l2) const throw(bad_parameter) {
#ifdef LIBTENSOR_DEBUG
    if (! is_valid(l1) && ! is_valid(l2))
        throw bad_parameter(g_ns, k_clazz, "product(label_t, label_t) const",
                __FILE__, __LINE__, "Invalid label.");
#endif

    return (l1 < l2 ? determine_product(l1, l2) : determine_product(l2, l1));
}

product_table_i::label_set_t product_table_i::product(label_t l1,
        const label_set_t &l2) const throw(bad_parameter) {
#ifdef LIBTENSOR_DEBUG
    static const char *method = "product(label_t, const label_set_t &) const";

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
        label_set_t lsx = (*it < l1 ?
                determine_product(*it, l1) :
                determine_product(l1, *it));
        ls.insert(lsx.begin(), lsx.end());
    }
    return ls;
}

product_table_i::label_set_t product_table_i::product(const label_set_t &l1,
        label_t l2) const throw(bad_parameter) {
#ifdef LIBTENSOR_DEBUG
    static const char *method = "product(const label_set_t &, label_t) const";

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
        label_set_t lsx = (*it < l2 ?
                determine_product(*it, l2) :
                determine_product(l2, *it));
        ls.insert(lsx.begin(), lsx.end());
    }
    return ls;
}

product_table_i::label_set_t product_table_i::product(const label_set_t &l1,
        const label_set_t &l2) const throw(bad_parameter){

#ifdef LIBTENSOR_DEBUG
    static const char *method = "product(label_set_t, label_set_t)";

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
            label_set_t lsx = (*it1 < *it2 ?
                    determine_product(*it1, *it2) :
                    determine_product(*it2, *it1));
            ls.insert(lsx.begin(), lsx.end());
        }
    }

    return ls;
}

product_table_i::label_set_t product_table_i::product(
        const label_group_t &lg) const {

    if (lg.empty()) return label_set_t();

    label_group_t::const_iterator it = lg.begin();
    label_set_t ls;
    ls.insert(*it);
    it++;
    for (; it != lg.end(); it++) ls = product(*it, ls);

    return ls;
}

bool product_table_i::is_in_product(const label_group_t &lg,
        label_t l) const throw(bad_parameter) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
            "is_in_product(const label_group_t &, label_t)";
    if (! is_valid(l))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid irrep l.");
#endif

    if (lg.empty()) return false;
    return (product(lg).count(l) != 0);
}

void product_table_i::check() const throw(bad_symmetry) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "check() const";

    // Check identity label products
    for (label_t l = 0; l < get_n_labels(); l++) {

        label_set_t ls = determine_product(k_identity, l);
        if (ls.size() != 1) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Size of product with identity.");
        }
        if (*(ls.begin()) != l) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Result of product with identity.");
        }

        for (label_t ll = l; ll < get_n_labels(); ll++) {
            label_set_t lls = determine_product(l, ll);
            if (lls.empty()) {
                throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
                        __LINE__, "Product table not properly setup.");
            }
        }
    }
#endif

    do_check();
}

} // namespace libtensor


