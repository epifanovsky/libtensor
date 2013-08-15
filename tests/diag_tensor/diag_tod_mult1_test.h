#ifndef LIBTENSOR_DIAG_TOD_MULT1_TEST_H
#define LIBTENSOR_DIAG_TOD_MULT1_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_mult1 class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_mult1_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_1(bool recip, bool zero, size_t ni);
    void test_ij_2(bool recip, bool zero, size_t ni);
    void test_ij_3(bool recip, bool zero, size_t ni);
    void test_ij_4(bool recip, bool zero, size_t ni);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_MULT1_TEST_H

