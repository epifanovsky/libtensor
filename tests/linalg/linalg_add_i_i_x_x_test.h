#ifndef LIBTENSOR_LINALG_ADD_I_I_X_X_TEST_H
#define LIBTENSOR_LINALG_ADD_I_I_X_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (add_i_i_x_x)

    \ingroup libtensor_tests_linalg
 **/
class linalg_add_i_i_x_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_add_i_i_x_x(size_t ni, size_t sia, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADD_I_I_X_X_TEST_H
