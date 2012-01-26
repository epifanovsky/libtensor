#include <sstream>
#include <vector>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_i_i_i_test.h"

namespace libtensor {


void linalg_i_i_i_test::perform() throw(libtest::test_exception) {

    test_i_i_i(1, 1, 1, 1);
    test_i_i_i(2, 1, 1, 1);
    test_i_i_i(16, 1, 1, 1);
    test_i_i_i(17, 1, 1, 1);
    test_i_i_i(1024, 1, 1, 1);
    test_i_i_i(1031, 1, 1, 1);
    test_i_i_i(2, 2, 3, 1);
    test_i_i_i(2, 3, 1, 2);
    test_i_i_i(300, 5, 6, 7);
}


void linalg_i_i_i_test::test_i_i_i(size_t ni, size_t sia, size_t sib,
    size_t sic) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_i_i_i_test::test_i_i_i("
        << ni << ", " << sia << ", " << sib << ", " << sic << ")";
    std::string tnss = ss.str();

    try {

    size_t sza = ni * sia, szb = ni * sib, szc = ni * sic;

    std::vector<double> a(sza, 0.0), b(szb, 0.0), c(szc, 0.0), c_ref(szc, 0.0);

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    linalg::i_i_i(ni, &a[0], sia, &b[0], sib, &c[0], sic);
    linalg_base_generic::i_i_i(ni, &a[0], sia, &b[0], sib, &c_ref[0], sic);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result.");
        }
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
