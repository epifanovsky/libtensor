#include <sstream>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_add_i_i_x_x_test.h"

namespace libtensor {


void linalg_add_i_i_x_x_test::perform() throw(libtest::test_exception) {

    test_add_i_i_x_x(1, 1, 1);
    test_add_i_i_x_x(2, 1, 1);
    test_add_i_i_x_x(16, 1, 1);
    test_add_i_i_x_x(17, 1, 1);
    test_add_i_i_x_x(2, 2, 3);
    test_add_i_i_x_x(2, 3, 2);
}


void linalg_add_i_i_x_x_test::test_add_i_i_x_x(size_t ni, size_t sia,
    size_t sic) {

    std::ostringstream ss;
    ss << "linalg_add_i_i_x_x_test::test_add_i_i_x_x("
        << ni << ", " << sia << ", " << sic << ")";
    std::string tnss = ss.str();

    try {

    size_t sza = ni * sia, szc = ni * sic;
    std::vector<double> a(sza, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double b = 0.0, d = 0.0;

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    double ka = drand48() - 0.5;
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();
    b = drand48();
    double kb = drand48() - 0.5;
    d = drand48();

    linalg::add_i_i_x_x(0, ni, &a[0], sia, ka, b, kb, &c[0], sic, d);
    linalg_generic::add_i_i_x_x(0, ni, &a[0], sia, ka, b, kb, &c_ref[0],
        sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
        }
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
