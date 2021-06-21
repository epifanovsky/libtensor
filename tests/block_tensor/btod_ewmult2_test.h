#ifndef LIBTENSOR_BTOD_EWMULT2_TEST_H
#define LIBTENSOR_BTOD_EWMULT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_ewmult2 class

    \ingroup libtensor_tests
 **/
class btod_ewmult2_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1(bool doadd);
    void test_2(bool doadd);
    void test_3(bool doadd);
    void test_4(bool doadd);
    void test_5(bool doadd);
    void test_6(bool doadd);
    void test_7();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_TEST_H
