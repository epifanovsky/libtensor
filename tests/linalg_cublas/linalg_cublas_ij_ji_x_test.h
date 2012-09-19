#ifndef LIBTENSOR_LINALG_CUBLAS_IJ_JI_X_TEST_H
#define LIBTENSOR_LINALG_CUBLAS_IJ_JI_X_TEST_H

#include "../linalg/linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg_cublas class (ij_ji_x)

    \ingroup libtensor_tests_linalg_cublas
 **/
class linalg_cublas_ij_ji_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_ji_x(size_t ni, size_t nj, size_t sja, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_IJ_JI_X_TEST_H
