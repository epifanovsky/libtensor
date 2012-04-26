#ifndef LIBTENSOR_LINALG_IJ_PI_JP_X_TEST_H
#define LIBTENSOR_LINALG_IJ_PI_JP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (ij_pi_jp_x)

    \ingroup libtensor_tests_linalg
 **/
class linalg_ij_pi_jp_x_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_pi_jp_x(size_t ni, size_t nj, size_t np, size_t sic,
        size_t sjb, size_t spa) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJ_PI_JP_X_TEST_H
