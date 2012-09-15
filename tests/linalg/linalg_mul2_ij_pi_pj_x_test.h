#ifndef LIBTENSOR_LINALG_MUL2_IJ_PI_PJ_X_TEST_H
#define LIBTENSOR_LINALG_MUL2_IJ_PI_PJ_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (mul2_ij_pi_pj_x)

    \ingroup libtensor_tests_linalg
 **/
class linalg_mul2_ij_pi_pj_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_mul2_ij_pi_pj_x(size_t ni, size_t nj, size_t np, size_t sic,
        size_t spa, size_t spb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MUL2_IJ_PI_PJ_X_TEST_H
