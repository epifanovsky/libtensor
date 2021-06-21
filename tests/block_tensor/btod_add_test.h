#ifndef LIBTENSOR_BTOD_ADD_TEST_H
#define LIBTENSOR_BTOD_ADD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_add class

    \ingroup libtensor_tests_btod
 **/
class btod_add_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1(double ca1, double ca2);
    void test_2(double ca1, double ca2, double cs)
       ;
    void test_3(double ca1, double ca2);
    void test_4(double ca1, double ca2, double ca3, double ca4)
       ;
    void test_5();
    void test_6();
    void test_7();
    void test_8();
    void test_9();
    void test_10(double d);
    void test_11(bool sign);

    /** \brief Tests if exceptions are thrown when the tensors have
            different dimensions
    **/
    void test_exc();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_TEST_H
