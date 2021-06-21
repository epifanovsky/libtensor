#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_TEST_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_symmetrize3 class

    \ingroup libtensor_tests_btod
 **/
class btod_symmetrize3_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8a();
    void test_8b();
    void test_9();
    void test_10();

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_TEST_H
