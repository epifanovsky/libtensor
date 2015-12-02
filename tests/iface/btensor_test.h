#ifndef LIBTENSOR_BTENSOR_TEST_H
#define LIBTENSOR_BTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btensor class

    \ingroup libtensor_tests_iface
**/
class btensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_re_1();
    void test_re_2();
    void test_cx_1();
    void test_cx_2();

};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_TEST_H

