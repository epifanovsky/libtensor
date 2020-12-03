#include "test_utils.h"
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/linalg/linalg_generic.h>
#include <sstream>
#include <vector>

using namespace libtensor;

int test_mul2_i_i_x(size_t ni, size_t sia, size_t sic) {

  std::ostringstream ss;
  ss << "test_mul2_i_i_x(" << ni << ", " << sia << ", " << sic << ")";
  std::string tnss = ss.str();

  try {

    size_t sza = ni * sia, szc = ni * sic;
    std::vector<double> a(sza, 0.0), c(szc, 0.0), c_ref(szc, 0.0);
    double b = 0.0;

    for (size_t i = 0; i < sza; i++) a[i] = drand48();
    for (size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();
    b = drand48();

    linalg::mul2_i_i_x(0, ni, &a[0], sia, b, &c[0], sic);
    linalg_generic::mul2_i_i_x(0, ni, &a[0], sia, b, &c_ref[0], sic);

    for (size_t i = 0; i < szc; i++) {
      if (!cmp(c[i] - c_ref[i], c_ref[i])) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
      }
    }

  } catch (exception& e) {
    return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
  }

  return 0;
}

int main() {

  return

        test_mul2_i_i_x(1, 1, 1) | test_mul2_i_i_x(2, 1, 1) | test_mul2_i_i_x(16, 1, 1) |
        test_mul2_i_i_x(17, 1, 1) | test_mul2_i_i_x(2, 2, 3) | test_mul2_i_i_x(2, 3, 2) |

        0;
}

