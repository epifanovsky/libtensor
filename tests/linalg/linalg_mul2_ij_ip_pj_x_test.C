#include "test_utils.h"
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/linalg_generic.h>
#include <sstream>
#include <vector>

using namespace libtensor;

int test_mul2_ij_ip_pj_x(size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
                         size_t spb) {

  std::ostringstream ss;
  ss << "test_mul2_ij_ip_pj_x(" << ni << ", " << nj << ", " << np << ", " << sia << ", "
     << sic << ", " << spb << ")";
  std::string tnss = ss.str();

  try {

    size_t sza = ni * sia, szb = np * spb, szc = ni * sic;
    std::vector<double> a(sza, 0.0), b(szb, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double d = 0.0;

    for (size_t i = 0; i < sza; i++) a[i] = drand48();
    for (size_t i = 0; i < szb; i++) b[i] = drand48();
    for (size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c_ref[0], sic,
                                    d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = 0.0).");
      }
    }

    d = 1.0;
    linalg::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c_ref[0], sic,
                                    d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = 1.0).");
      }
    }

    d = -1.0;
    linalg::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c_ref[0], sic,
                                    d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__,
                         "Incorrect result (d = -1.0).");
      }
    }

    d = drand48();
    linalg::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c_ref[0], sic,
                                    d);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result (d = rnd).");
      }
    }

    d = -drand48();
    linalg::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c[0], sic, d);
    linalg_generic::mul2_ij_ip_pj_x(0, ni, nj, np, &a[0], sia, &b[0], spb, &c_ref[0], sic,
                                    d);

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

        test_mul2_ij_ip_pj_x(1, 1, 1, 1, 1, 1) | test_mul2_ij_ip_pj_x(1, 2, 3, 3, 2, 2) |
        test_mul2_ij_ip_pj_x(2, 1, 3, 3, 1, 1) |
        test_mul2_ij_ip_pj_x(16, 16, 1, 1, 16, 16) |
        test_mul2_ij_ip_pj_x(3, 17, 2, 2, 17, 17) |
        test_mul2_ij_ip_pj_x(2, 2, 2, 2, 3, 4) | test_mul2_ij_ip_pj_x(2, 2, 2, 4, 3, 2) |

        0;
}

