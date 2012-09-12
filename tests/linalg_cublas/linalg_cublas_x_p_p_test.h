#ifndef LIBTENSOR_LINALG_CUBLAS_X_P_P_TEST_H
#define LIBTENSOR_LINALG_CUBLAS_X_P_P_TEST_H

#include "../linalg/linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg_cublas class (x_p_p)

    \ingroup libtensor_tests_linalg_cublas
 **/
class linalg_cublas_x_p_p_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_x_p_p(size_t np, size_t spa, size_t spb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_X_P_P_TEST_H
