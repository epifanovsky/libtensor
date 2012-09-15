#ifndef LIBTENSOR_LINALG_X_P_P_TEST_H
#define LIBTENSOR_LINALG_X_P_P_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (mul2_x_p_p)

    \ingroup libtensor_tests_linalg
 **/
class linalg_x_p_p_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_x_p_p(size_t np, size_t spa, size_t spb)
        throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_X_P_P_TEST_H
