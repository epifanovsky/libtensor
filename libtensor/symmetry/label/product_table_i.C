#include "product_table_i.h"

namespace libtensor {

const char *product_table_i::k_clazz = "product_table_i";
const product_table_i::label_t product_table_i::k_invalid =
        (product_table_i::label_t) -1;

void product_table_i::check() const throw(generic_exception) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "check()";

    label_t lx = get_identity();
    label_set_t ls = get_complete_set();
    for (label_set_t::const_iterator it1 = ls.begin();
            it1 != ls.end(); it1++) {

        // Check symmetry of table
        for (label_set_t::const_iterator it2 = ls.begin(); it2 != it1; it2++) {

            label_set_t lij = product(*it1, *it2);
            label_set_t lji = product(*it2, *it1);

            if (lij.size() != lji.size())
                throw generic_exception(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Non-symmetric product table.");

            label_set_t::const_iterator iij = lij.begin(),
                    iji = lji.begin();
            for (; iij != lij.end() && *iij == *iji; iij++, iji++);

            if (iij != lij.end())
                throw generic_exception(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Non-symmetric product table.");
        }

        // Check if the identity label fulfills the conditions
        label_set_t lr1 = product(lx, *it1);
        if (lr1.size() != 1)
            throw generic_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Product with identity.");

        if (*(lr1.begin()) != *it1)
            throw generic_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Product with identity.");

        label_set_t lr2 = product(*it1, *it1);
        if (lr2.count(lx) == 0)
            throw generic_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Product with self.");

    }

#endif
}

} // namespace libtensor


