#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ij_ipq_jqp_x_test.h"

namespace libtensor {


void linalg_ij_ipq_jqp_x_test::perform() throw(libtest::test_exception) {

    //                ni  nj  np  nq  sia sic sjb spa sqb
    test_ij_ipq_jqp_x(1,  1,  1,  1,  1,  1,  1,  1,  1);
    test_ij_ipq_jqp_x(1,  1,  2,  1,  2,  1,  2,  1,  2);
    test_ij_ipq_jqp_x(1,  2,  1,  2,  2,  2,  2,  2,  1);
    test_ij_ipq_jqp_x(2,  1,  1,  1,  1,  1,  1,  1,  1);
    test_ij_ipq_jqp_x(2,  2,  2,  2,  4,  2,  4,  2,  2);
    test_ij_ipq_jqp_x(5,  3,  7,  4,  28, 3,  28, 4,  7);
    test_ij_ipq_jqp_x(16, 16, 16, 16, 256,16, 256,16, 16);
    test_ij_ipq_jqp_x(17, 9,  5,  2,  60, 10, 30, 10, 10);
}


void linalg_ij_ipq_jqp_x_test::test_ij_ipq_jqp_x(size_t ni, size_t nj,
    size_t np, size_t nq, size_t sia, size_t sic, size_t sjb, size_t spa,
    size_t sqb) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ij_ipq_jqp_x_test::test_ij_ipq_jqp_x("
        << ni << ", " << nj << ", " << np << ", " << nq << ", "
        << sia << ", " << sic << ", " << sjb << ", " << spa << ", "
        << sqb << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
        d);
    linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
        sjb, c_ref, sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
        d);
    linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
        sjb, c_ref, sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
        d);
    linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
        sjb, c_ref, sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
        d);
    linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
        sjb, c_ref, sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
        d);
    linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
        sjb, c_ref, sic, d);

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
