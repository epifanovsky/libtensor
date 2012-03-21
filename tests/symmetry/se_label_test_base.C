#include <sstream>
#include <libtensor/btod/transf_double.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include "se_label_test.h"

namespace libtensor {

std::string se_label_test_base::setup_pg_table() throw(libtest::test_exception){

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
        point_group_table s6("s6", im, "Ag");
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
        fail_test("se_label_test_base::setup_s6_symmetry()",
                __FILE__, __LINE__, e.what());
    }

    return "s6";
}


} // namespace libtensor
