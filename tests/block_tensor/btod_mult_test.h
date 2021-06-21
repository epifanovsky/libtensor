#ifndef LIBTENSOR_BTOD_MULT_TEST_H
#define LIBTENSOR_BTOD_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_mult class

    \ingroup libtensor_tests_btod
**/
class btod_mult_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1(bool recip, bool doadd);
    void test_2(bool recip, bool doadd);
    void test_3(bool recip, bool doadd);
    void test_4(bool recip, bool doadd);
    void test_5(bool symm1, bool symm2);
    void test_6(bool symm1, bool symm2);
    void test_7(bool label, bool part, bool asymm,
            bool recip, bool add);
    void test_8a(bool label, bool part);
    void test_8b(bool label, bool part);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_TEST_H
