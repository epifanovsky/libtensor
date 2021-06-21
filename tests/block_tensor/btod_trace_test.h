#ifndef LIBTENSOR_BTOD_TRACE_TEST_H
#define LIBTENSOR_BTOD_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_trace class

    \ingroup libtensor_tests_btod
 **/
class btod_trace_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_zero_1();
    void test_nosym_1();
    void test_nosym_1_sp();
    void test_nosym_2();
    void test_nosym_3();
    void test_nosym_4();
    void test_nosym_5();
    void test_nosym_6();
    void test_nosym_7();
    void test_permsym_1();
    void test_permsym_2();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRACE_TEST_H
