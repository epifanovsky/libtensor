#ifndef LIBTENSOR_ORBIT_TEST_H
#define LIBTENSOR_ORBIT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::orbit class

    \ingroup libtensor_tests_core
**/
class orbit_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8();
    void test_9();
    void test_10();
    void test_11();

};


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_TEST_H
