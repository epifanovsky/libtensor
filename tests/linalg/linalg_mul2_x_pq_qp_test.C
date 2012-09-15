#include <sstream>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_mul2_x_pq_qp_test.h"

namespace libtensor {


void linalg_mul2_x_pq_qp_test::perform() throw(libtest::test_exception) {

    test_mul2_x_pq_qp(1, 1, 1, 1);
    test_mul2_x_pq_qp(1, 2, 2, 1);
    test_mul2_x_pq_qp(2, 1, 1, 2);
    test_mul2_x_pq_qp(2, 2, 2, 2);
    test_mul2_x_pq_qp(16, 2, 2, 16);
    test_mul2_x_pq_qp(17, 3, 3, 17);
    test_mul2_x_pq_qp(2, 2, 3, 4);
    test_mul2_x_pq_qp(2, 2, 4, 3);
}


void linalg_mul2_x_pq_qp_test::test_mul2_x_pq_qp(size_t np, size_t nq,
    size_t spa, size_t sqb) {

    std::ostringstream ss;
    ss << "linalg_mul2_x_pq_qp_test::test_mul2_x_pq_qp("
        << np << ", " << nq << ", " << spa << ", " << sqb << ")";
    std::string tnss = ss.str();

    try {

    size_t sza = np * spa, szb = nq * sqb;

    std::vector<double> a(sza, 0.0), b(szb, 0.0);

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();

    double c = linalg::mul2_x_pq_qp(np, nq, &a[0], spa, &b[0], sqb);
    double c_ref = linalg_generic::mul2_x_pq_qp(np, nq, &a[0], spa, &b[0], sqb);

    if(!cmp(c - c_ref, c_ref)) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
