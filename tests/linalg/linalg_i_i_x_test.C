#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_i_i_x_test.h"

namespace libtensor {


void linalg_i_i_x_test::perform() throw(libtest::test_exception) {

    test_i_i_x(1, 1, 1);
    test_i_i_x(2, 1, 1);
    test_i_i_x(16, 1, 1);
    test_i_i_x(17, 1, 1);
    test_i_i_x(2, 2, 3);
    test_i_i_x(2, 3, 2);
}


void linalg_i_i_x_test::test_i_i_x(size_t ni, size_t sia, size_t sic)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_i_i_x_test::test_i_i_x("
        << ni << ", " << sia << ", " << sic << ")";
    std::string tnss = ss.str();

    double *a = 0, *c = 0, *c_ref = 0;
    double b = 0.0;

    try {

    size_t sza = ni * sia, szc = ni * sic;

    a = new double[sza];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();
    b = drand48();

    linalg::i_i_x(ni, a, sia, b, c, sic);
    linalg_base_generic::i_i_x(ni, a, sia, b, c_ref, sic);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result.");
        }
    }

    delete [] a; a = 0;
    delete [] c; c = 0;
    delete [] c_ref; c_ref = 0;

    } catch(exception &e) {
        delete [] a; a = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        delete [] a; a = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        throw;
    }
}


} // namespace libtensor
