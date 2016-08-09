#include <sstream>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "test_utils.h"

using namespace libtensor;


int test_mul2_ij_i_j_x(
    size_t ni, size_t nj, size_t sia, size_t sic, size_t sjb) {

    std::ostringstream ss;
    ss << "test_mul2_ij_i_j_x(" << ni << ", " << nj << ", " << sia << ", "
        << sic << ", " << sjb << ")";
    std::string tnss = ss.str();

    try {

    size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;
    std::vector<double> a(sza, 0.0), b(szb, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double d = 0.0;

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb, &c[0], sic, d);
    linalg_generic::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            return fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 0.0).");
        }
    }

    d = 1.0;
    linalg::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb, &c[0], sic, d);
    linalg_generic::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            return fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = 1.0).");
        }
    }

    d = -1.0;
    linalg::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb, &c[0], sic, d);
    linalg_generic::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            return fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -1.0).");
        }
    }

    d = drand48();
    linalg::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb, &c[0], sic, d);
    linalg_generic::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            return fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = rnd).");
        }
    }

    d = -drand48();
    linalg::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb, &c[0], sic, d);
    linalg_generic::mul2_ij_i_j_x(0, ni, nj, &a[0], sia, &b[0], sjb,
        &c_ref[0], sic, d);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            return fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result (d = -rnd).");
        }
    }

    } catch(exception &e) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main () {

    return

    test_mul2_ij_i_j_x(1, 1, 1, 1, 1) |
    test_mul2_ij_i_j_x(1, 2, 1, 2, 1) |
    test_mul2_ij_i_j_x(2, 1, 1, 1, 1) |
    test_mul2_ij_i_j_x(16, 16, 1, 16, 1) |
    test_mul2_ij_i_j_x(3, 17, 1, 17, 1) |
    test_mul2_ij_i_j_x(2, 2, 2, 3, 4) |
    test_mul2_ij_i_j_x(2, 2, 4, 3, 2) |

    0;
}

