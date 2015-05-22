#ifndef LIBTENSOR_SET_TEST_H
#define LIBTENSOR_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::set function

    \ingroup libtensor_tests_iface
 **/
class set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_s_1() throw(libtest::test_exception);
    void test_s_2() throw(libtest::test_exception);
    void test_d_1() throw(libtest::test_exception);
    void test_d_2() throw(libtest::test_exception);
    void test_x_1() throw(libtest::test_exception);
    void test_x_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SET_TEST_H
