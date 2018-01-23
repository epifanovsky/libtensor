#ifndef LIBTENSOR_LINALG_BLAS_VERSION_TEST_H
#define LIBTENSOR_LINALG_BLAS_VERSION_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (add_i_i_x_x)

    \ingroup libtensor_tests_linalg
 **/
class linalg_blas_version_test: public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BLAS_VERSION_TEST_H
