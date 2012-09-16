#ifndef LIBTENSOR_LINALG_MUL2_IJ_I_J_X_TEST_H
#define LIBTENSOR_LINALG_MUL2_IJ_I_J_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (mul2_ij_i_j_x)

    \ingroup libtensor_tests_linalg
 **/
class linalg_mul2_ij_i_j_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_mul2_ij_i_j_x(size_t ni, size_t nj, size_t sia, size_t sic,
        size_t sjb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MUL2_IJ_I_J_X_TEST_H
