#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include "linalg_x_pq_qp_test.h"

namespace libtensor {


void linalg_x_pq_qp_test::perform() throw(libtest::test_exception) {

    test_x_pq_qp(1, 1, 1, 1);
    test_x_pq_qp(1, 2, 2, 1);
    test_x_pq_qp(2, 1, 1, 2);
    test_x_pq_qp(2, 2, 2, 2);
    test_x_pq_qp(16, 2, 2, 16);
    test_x_pq_qp(17, 3, 3, 17);
    test_x_pq_qp(2, 2, 3, 4);
    test_x_pq_qp(2, 2, 4, 3);
}


void linalg_x_pq_qp_test::test_x_pq_qp(size_t np, size_t nq, size_t spa, size_t sqb)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_x_pq_qp_test::test_x_pq_qp("
        << np << ", " << nq << ", " << spa << ", " << sqb << ")";
    std::string tnss = ss.str();

    double *a = 0, *b = 0;

    try {

    size_t sza = np * spa, szb = nq * sqb;

    a = new double[sza];
    b = new double[szb];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szb; i++) b[i] = drand48();

    double c = linalg::x_pq_qp(np, nq, a, spa, b, sqb);
    double c_ref = linalg_generic::x_pq_qp(np, nq, a, spa, b, sqb);

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
