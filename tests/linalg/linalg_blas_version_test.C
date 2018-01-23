#include <sstream>
#include <libtensor/linalg/linalg.h>
#include <libtensor/exception.h>
#include "linalg_blas_version_test.h"

namespace libtensor {


void linalg_blas_version_test::perform() throw(libtest::test_exception) {
    std::cout << "BLAS version: " << linalg::k_clazz << "     ";
}




} // namespace libtensor
