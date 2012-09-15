#include <sstream>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_mul2_i_pi_p_x_test.h"

namespace libtensor {


void linalg_mul2_i_pi_p_x_test::perform() throw(libtest::test_exception) {

    test_mul2_i_pi_p_x(1, 1, 1, 1, 1);
    test_mul2_i_pi_p_x(1, 2, 1, 1, 1);
    test_mul2_i_pi_p_x(2, 1, 1, 2, 1);
    test_mul2_i_pi_p_x(16, 16, 1, 16, 1);
    test_mul2_i_pi_p_x(17, 3, 1, 17, 1);
    test_mul2_i_pi_p_x(2, 2, 2, 3, 4);
    test_mul2_i_pi_p_x(2, 2, 4, 3, 2);
}


void linalg_mul2_i_pi_p_x_test::test_mul2_i_pi_p_x(size_t ni, size_t np,
    size_t sic, size_t spa, size_t spb) {

    std::ostringstream ss;
    ss << "linalg_mul2_i_pi_p_x_test::test_mul2_i_pi_p_x("
        << ni << ", " << np << ", " << sic << ", " << spa << ", " << spb << ")";
    std::string tnss = ss.str();

    try {

    size_t sza = np * spa, szb = np * spb, szc = ni * sic;
    std::vector<double> a(sza, 0.0), b(szb, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double d = 0.0;

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_i_pi_p_x(0, ni, np, &a[0], spa, &b[0], spb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -rnd).");
        }
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
