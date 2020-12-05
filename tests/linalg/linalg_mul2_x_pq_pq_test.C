#include "test_utils.h"
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/linalg_generic.h>
#include <sstream>
#include <vector>

using namespace libtensor;

int test_mul2_x_pq_pq(size_t np, size_t nq, size_t spa, size_t spb) {

  std::ostringstream ss;
  ss << "test_mul2_x_pq_pq(" << np << ", " << nq << ", " << spa << ", " << spb << ")";
  std::string tnss = ss.str();

  try {

    size_t sza = np * spa, szb = np * spb;
    std::vector<double> a(sza, 0.0), b(szb, 0.0);

    for (size_t i = 0; i < sza; i++) a[i] = drand48();
    for (size_t i = 0; i < szb; i++) b[i] = drand48();

    double c     = linalg::mul2_x_pq_pq(0, np, nq, &a[0], spa, &b[0], spb);
    double c_ref = linalg_generic::mul2_x_pq_pq(0, np, nq, &a[0], spa, &b[0], spb);

    if (!cmp(c - c_ref, c_ref)) {
      return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
    }

  } catch (exception& e) {
    return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
  }

  return 0;
}

int main() {

  return

        test_mul2_x_pq_pq(1, 1, 1, 1) | test_mul2_x_pq_pq(1, 2, 2, 2) |
        test_mul2_x_pq_pq(2, 1, 1, 2) | test_mul2_x_pq_pq(2, 2, 2, 2) |
        test_mul2_x_pq_pq(16, 2, 2, 16) | test_mul2_x_pq_pq(17, 3, 3, 17) |
        test_mul2_x_pq_pq(2, 2, 3, 4) | test_mul2_x_pq_pq(2, 2, 4, 3) |

        0;
}

