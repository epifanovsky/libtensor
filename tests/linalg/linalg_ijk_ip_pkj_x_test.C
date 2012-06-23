#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ijk_ip_pkj_x_test.h"

namespace libtensor {


void linalg_ijk_ip_pkj_x_test::perform() throw(libtest::test_exception) {

    //                ni  nj  nk  np  sjc
    test_ijk_ip_pkj_x(1,  1,  1,  1,  1);
    test_ijk_ip_pkj_x(2,  1,  1,  1,  1);
    test_ijk_ip_pkj_x(1,  2,  1,  1,  2);
    test_ijk_ip_pkj_x(1,  1,  2,  1,  2);
    test_ijk_ip_pkj_x(1,  1,  1,  1,  1);
    test_ijk_ip_pkj_x(1,  1,  1,  2,  2);
    test_ijk_ip_pkj_x(1,  1,  1,  1,  3);
    test_ijk_ip_pkj_x(2,  3,  2,  2,  2);
    test_ijk_ip_pkj_x(3,  5,  1,  13, 1);
    test_ijk_ip_pkj_x(16, 16, 16, 16, 18);
    test_ijk_ip_pkj_x(17, 16, 17, 17, 17);
}


void linalg_ijk_ip_pkj_x_test::test_ijk_ip_pkj_x(size_t ni, size_t nj,
    size_t nk, size_t np, size_t sjc) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ijk_ip_pkj_x_test::test_ijk_ip_pkj_x("
        << ni << ", " << nj << ", " << nk << ", " << np << ", "
        << sjc << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t sza = ni * np, szb = np * nk * nj, szc = ni * nj * sjc;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c, sjc, nj * sjc, d);
    linalg_base_generic::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c_ref, sjc, nj * sjc, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c, sjc, nj * sjc, d);
    linalg_base_generic::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c_ref, sjc, nj * sjc, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c, sjc, nj * sjc, d);
    linalg_base_generic::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c_ref, sjc, nj * sjc, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c, sjc, nj * sjc, d);
    linalg_base_generic::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c_ref, sjc, nj * sjc, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c, sjc, nj * sjc, d);
    linalg_base_generic::ijk_ip_pkj_x(ni, nj, nk, np, a, np, b, nj, nj * nk,
        c_ref, sjc, nj * sjc, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -rnd).");
        }
    }

    delete [] a; a = 0;
    delete [] b; b = 0;
    delete [] c; c = 0;
    delete [] c_ref; c_ref = 0;

    } catch(exception &e) {
        delete [] a; a = 0;
        delete [] b; b = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        delete [] a; a = 0;
        delete [] b; b = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        throw;
    }
}


} // namespace libtensor
