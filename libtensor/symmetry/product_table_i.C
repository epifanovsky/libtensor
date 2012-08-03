#include "product_table_i.h"


namespace libtensor {


const char *product_table_i::k_clazz = "product_table_i";


const product_table_i::label_t product_table_i::k_invalid =
        (product_table_i::label_t) -1;


const product_table_i::label_t product_table_i::k_identity = 0;


void product_table_i::check() const throw(bad_symmetry) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "check() const";

    // Check identity label products
    for (label_t l = 0; l < get_n_labels(); l++) {

        label_set_t ls(product(k_identity, l));
        if (ls.size() != 1) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Size of product with identity.");
        }
        if (*(ls.begin()) != l) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Result of product with identity.");
        }

        for (label_t ll = l; ll < get_n_labels(); ll++) {
            label_set_t lls(product(l, ll));
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


