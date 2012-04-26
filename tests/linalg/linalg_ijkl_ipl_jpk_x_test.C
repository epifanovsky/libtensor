#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ijkl_ipl_jpk_x_test.h"

namespace libtensor {


void linalg_ijkl_ipl_jpk_x_test::perform() throw(libtest::test_exception) {

    test_ijkl_ipl_jpk_x_1(1, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_1(2, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_1(1, 2, 1, 1, 1);
    test_ijkl_ipl_jpk_x_1(1, 1, 2, 1, 1);
    test_ijkl_ipl_jpk_x_1(1, 1, 1, 2, 1);
    test_ijkl_ipl_jpk_x_1(1, 1, 1, 1, 2);
    test_ijkl_ipl_jpk_x_1(2, 3, 2, 3, 2);
    test_ijkl_ipl_jpk_x_1(3, 5, 1, 7, 13);
    test_ijkl_ipl_jpk_x_1(16, 16, 16, 16, 16);
    test_ijkl_ipl_jpk_x_1(17, 16, 17, 16, 17);

    test_ijkl_ipl_jpk_x_2(1, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_2(2, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_2(1, 2, 1, 1, 1);
    test_ijkl_ipl_jpk_x_2(1, 1, 2, 1, 1);
    test_ijkl_ipl_jpk_x_2(1, 1, 1, 2, 1);
    test_ijkl_ipl_jpk_x_2(1, 1, 1, 1, 2);
    test_ijkl_ipl_jpk_x_2(2, 3, 2, 3, 2);
    test_ijkl_ipl_jpk_x_2(3, 5, 1, 7, 13);
    test_ijkl_ipl_jpk_x_2(16, 16, 16, 16, 16);
    test_ijkl_ipl_jpk_x_2(17, 16, 17, 16, 17);

    test_ijkl_ipl_jpk_x_3(1, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_3(2, 1, 1, 1, 1);
    test_ijkl_ipl_jpk_x_3(1, 2, 1, 1, 1);
    test_ijkl_ipl_jpk_x_3(1, 1, 2, 1, 1);
    test_ijkl_ipl_jpk_x_3(1, 1, 1, 2, 1);
    test_ijkl_ipl_jpk_x_3(1, 1, 1, 1, 2);
    test_ijkl_ipl_jpk_x_3(2, 3, 2, 3, 2);
    test_ijkl_ipl_jpk_x_3(3, 5, 1, 7, 13);
    test_ijkl_ipl_jpk_x_3(16, 16, 16, 16, 16);
    test_ijkl_ipl_jpk_x_3(17, 16, 17, 16, 17);
}


void linalg_ijkl_ipl_jpk_x_test::test_ijkl_ipl_jpk_x_1(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ijkl_ipl_kpj_x_test::test_ijkl_ipl_jpk_x_1("
        << ni << ", " << nj << ", " << nk << ", " << nl << ", "
        << np << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t spa = nl, sia = spa * np;
    size_t spb = nk, sjb = spb * np;
    size_t sza = ni * sia, szb = nj * sjb, szc = ni * nj * nk * nl;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

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


void linalg_ijkl_ipl_jpk_x_test::test_ijkl_ipl_jpk_x_2(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ijkl_ipl_kpj_x_test::test_ijkl_ipl_jpk_x_2("
        << ni << ", " << nj << ", " << nk << ", " << nl << ", "
        << np << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t spa = nl + 2, sia = spa * np;
    size_t spb = nk + 3, sjb = spb * np;
    size_t sza = ni * sia, szb = nj * sjb, szc = ni * nj * nk * nl;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

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


void linalg_ijkl_ipl_jpk_x_test::test_ijkl_ipl_jpk_x_3(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ijkl_ipl_kpj_x_test::test_ijkl_ipl_jpk_x_3("
        << ni << ", " << nj << ", " << nk << ", " << nl << ", "
        << np << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t spa = nl * 2, sia = spa * np + 4;
    size_t spb = nk * 3, sjb = spb * np + 10;
    size_t sza = ni * sia, szb = nj * sjb, szc = ni * nj * nk * nl;

    a = new double[sza];
    b = new double[szb];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b, spb, sjb, c,
        d);
    linalg_base_generic::ijkl_ipl_jpk_x(ni, nj, nk, nl, np, a, spa, sia, b,
        spb, sjb, c_ref, d);

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
