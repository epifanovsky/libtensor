#ifndef LIBTENSOR_BTOD_TRIDIAGONALIZE_TEST_H
#define LIBTENSOR_BTOD_TRIDIAGONALIZE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_tridiagonalize class

    \ingroup libtensor_tests_btod
**/
class btod_tridiagonalize_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRIDIAGONALIZE_TEST_H
