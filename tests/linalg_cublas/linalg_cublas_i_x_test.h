#ifndef LIBTENSOR_LINALG_CUBLAS_I_X_TEST_H
#define LIBTENSOR_LINALG_CUBLAS_I_X_TEST_H

#include "../linalg/linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg_cublas class (i_x)

    \ingroup libtensor_tests_linalg_cublas
 **/
class linalg_cublas_i_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_i_x(size_t ni, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_I_X_TEST_H
