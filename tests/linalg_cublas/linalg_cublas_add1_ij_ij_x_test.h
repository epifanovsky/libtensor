#ifndef LIBTENSOR_LINALG_CUBLAS_ADD1_IJ_IJ_X_TEST_H
#define LIBTENSOR_LINALG_CUBLAS_ADD1_IJ_IJ_X_TEST_H

#include "../linalg/linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg_cublas class (add1_ij_ij_x)

    \ingroup libtensor_tests_linalg_cublas
 **/
class linalg_cublas_add1_ij_ij_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_add1_ij_ij_x(size_t ni, size_t nj, size_t sia, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_ADD1_IJ_IJ_X_TEST_H
