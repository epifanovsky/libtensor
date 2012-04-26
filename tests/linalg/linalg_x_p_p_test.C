#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_x_p_p_test.h"

namespace libtensor {


void linalg_x_p_p_test::perform() throw(libtest::test_exception) {

    test_x_p_p(1, 1, 1);
    test_x_p_p(2, 1, 1);
    test_x_p_p(16, 1, 1);
    test_x_p_p(17, 1, 1);
    test_x_p_p(2, 2, 3);
    test_x_p_p(2, 3, 2);
}


void linalg_x_p_p_test::test_x_p_p(size_t np, size_t spa, size_t spb)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_x_p_p_test::test_x_p_p("
        << np << ", " << spa << ", " << spb << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0;

    try {

    size_t sza = np * spa, szb = np * spb;

    a = new double[sza];
    b = new double[szb];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();

    double c = linalg::x_p_p(np, a, spa, b, spb);
    double c_ref = linalg_base_generic::x_p_p(np, a, spa, b, spb);

    if(!cmp(c - c_ref, c_ref)) {
        fail_test(tnss.c_str(), __FILE__, __LINE__,
            "Incorrect result.");
    }

    delete [] a; a = 0;
    delete [] b; b = 0;

    } catch(exception &e) {
        delete [] a;
        delete [] b;
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        delete [] a;
        delete [] b;
        throw;
    }
}


} // namespace libtensor
