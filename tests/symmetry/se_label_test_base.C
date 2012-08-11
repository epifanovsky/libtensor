#include <sstream>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include "se_label_test.h"

namespace libtensor {

void se_label_test_base::setup_pg_table(
        const std::string &id) throw(libtest::test_exception){

    if (id == "C2v" || id == "c2v") {

        try {

            // C\f$_{2v}\f$ point group - irreps: A1, A2, B1, B2
            // Product table:
            //      A1   A2   B1   B2
            // A1   A1   A2   B1   B2
            // A2   A2   A1   B2   B1
            // B1   B1   B2   A1   A2
            // B2   B2   B1   A2   A1
            point_group_table::label_t a1 = 0, a2 = 1, b1 = 2, b2 = 3;
            std::vector<std::string> im(4);
            im[a1] = "A1"; im[a2] = "A2"; im[b1] = "B1"; im[b2] = "B2";
            point_group_table c2v(id, im, "A1");
            c2v.add_product(a2, a2, a1);
            c2v.add_product(a2, b1, b2);
            c2v.add_product(a2, b2, b1);
            c2v.add_product(b1, b1, a1);
            c2v.add_product(b1, b2, a2);
            c2v.add_product(b2, b2, a1);
            c2v.check();
            product_table_container::get_instance().add(c2v);

        } catch (exception &e) {
            fail_test("se_label_test_base::setup_pg_table()",
                    __FILE__, __LINE__, e.what());
        }

    }
    else if (id == "S6" || id == "s6") {

        try {

            // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
            // Product table:
            //      Ag   Eg      Au   Eu
            // Ag   Ag   Eg      Au   Eu
            // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
            // Au   Au   Eu      Ag   Eg
            // Eu   Eu   2Au+Eu  Eg   2Ag+Eu
            point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
            std::vector<std::string> im(4);
            im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
            point_group_table s6(id, im, "Ag");
            s6.add_product(eg, eg, ag);
            s6.add_product(eg, eg, eg);
            s6.add_product(eg, au, eu);
            s6.add_product(eg, eu, au);
            s6.add_product(eg, eu, eu);
            s6.add_product(au, au, ag);
            s6.add_product(au, eu, eg);
            s6.add_product(eu, eu, ag);
            s6.add_product(eu, eu, eg);
            s6.check();
            product_table_container::get_instance().add(s6);

        } catch (exception &e) {
            fail_test("se_label_test_base::setup_pg_table()",
                    __FILE__, __LINE__, e.what());
        }
    }
    else {
        fail_test("se_label_test_base::setup_pg_table()",
                __FILE__, __LINE__, "Unknown ID.");
    }
}

void se_label_test_base::clear_pg_table(
        const std::string &id) throw(libtest::test_exception){

    if (product_table_container::get_instance().table_exists(id)) {
        product_table_container::get_instance().erase(id);
    }
}

} // namespace libtensor
