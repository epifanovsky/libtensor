#include "test_utils.h"
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/linalg_generic.h>
#include <sstream>
#include <vector>

using namespace libtensor;

int test_mul2_i_ipq_qp_x(size_t ni, size_t np, size_t nq, size_t sia, size_t sic,
                         size_t spa, size_t sqb) {

  std::ostringstream ss;
  ss << "test_mul2_i_ipq_qp_x(" << ni << ", " << np << ", " << nq << ", " << sia << ", "
     << sic << ", " << spa << ", " << sqb << ")";
  std::string tnss = ss.str();

  try {

    size_t sza = ni * sia, szb = nq * sqb, szc = ni * sic;
    std::vector<double> a(sza, 0.0), b(szb, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double d = 0.0;

    for (size_t i = 0; i < sza; i++) a[i] = drand48();
    for (size_t i = 0; i < szb; i++) b[i] = drand48();
    for (size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c[0], sic, d);
    linalg_generic::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c_ref[0],
                                    sic, d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = 0.0).");
      }
    }

    d = 1.0;
    linalg::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c[0], sic, d);
    linalg_generic::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c_ref[0],
                                    sic, d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = 1.0).");
      }
    }

    d = -1.0;
    linalg::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c[0], sic, d);
    linalg_generic::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c_ref[0],
                                    sic, d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__,
                         "Incorrect result (d = -1.0).");
      }
    }

    d = drand48();
    linalg::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c[0], sic, d);
    linalg_generic::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c_ref[0],
                                    sic, d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = rnd).");
      }
    }

    d = -drand48();
    linalg::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c[0], sic, d);
    linalg_generic::mul2_i_ipq_qp_x(0, ni, np, nq, &a[0], spa, sia, &b[0], sqb, &c_ref[0],
                                    sic, d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__,
                         "Incorrect result (d = -rnd).");
      }
    }

  } catch (exception& e) {
    return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
  }

  return 0;
}

int main() {

  return

        //                   ni  np  nq  sia sic spa sqb
        test_mul2_i_ipq_qp_x(1, 1, 1, 1, 1, 1, 1) |
        test_mul2_i_ipq_qp_x(1, 1, 2, 2, 1, 2, 1) |
        test_mul2_i_ipq_qp_x(1, 2, 1, 2, 1, 1, 2) |
        test_mul2_i_ipq_qp_x(2, 1, 1, 1, 2, 1, 1) |
        test_mul2_i_ipq_qp_x(2, 2, 2, 4, 2, 2, 2) |
        test_mul2_i_ipq_qp_x(5, 3, 7, 21, 5, 7, 3) |
        test_mul2_i_ipq_qp_x(16, 16, 16, 256, 16, 16, 16) |
        test_mul2_i_ipq_qp_x(17, 9, 5, 50, 20, 5, 10) |

        0;
}

