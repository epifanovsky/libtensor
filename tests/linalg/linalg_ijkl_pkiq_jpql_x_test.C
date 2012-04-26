#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ijkl_pkiq_jpql_x_test.h"

namespace libtensor {


void linalg_ijkl_pkiq_jpql_x_test::perform() throw(libtest::test_exception) {

    test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 1, 1);
    test_ijkl_pkiq_jpql_x(2, 1, 1, 1, 1, 1);
    test_ijkl_pkiq_jpql_x(1, 2, 1, 1, 1, 1);
    test_ijkl_pkiq_jpql_x(1, 1, 2, 1, 1, 1);
    test_ijkl_pkiq_jpql_x(1, 1, 1, 2, 1, 1);
    test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 2, 1);
    test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 1, 2);
    test_ijkl_pkiq_jpql_x(2, 3, 2, 3, 2, 3);
    test_ijkl_pkiq_jpql_x(3, 5, 1, 7, 13, 11);
    test_ijkl_pkiq_jpql_x(16, 16, 16, 16, 16, 16);
    test_ijkl_pkiq_jpql_x(17, 16, 17, 16, 17, 16);
}


void linalg_ijkl_pkiq_jpql_x_test::test_ijkl_pkiq_jpql_x(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np, size_t nq)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ijkl_pkiq_jpql_x_test::test_ijkl_pkiq_jpql_x("
        << ni << ", " << nj << ", " << nk << ", " << nl << ", "
        << np << ", " << nq << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t sza = np * nk * ni * nq, szb = nj * np * nq * nl,
        szc = ni * nj * nk * nl;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
    linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
        a, b, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
    linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
        a, b, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
    linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
        a, b, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
    linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
        a, b, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
    linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
        a, b, c_ref, d);

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
