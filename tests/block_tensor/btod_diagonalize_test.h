#ifndef LIBTENSOR_BTOD_DIAGONALIZE_TEST_H
#define LIBTENSOR_BTOD_DIAGONALIZE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_diagonalize class

    \ingroup libtensor_tests_btod
**/
class btod_diagonalize_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAGONALIZE_TEST_H
