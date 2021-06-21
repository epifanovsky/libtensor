#ifndef LIBTENSOR_BTOD_SET_ELEM_TEST_H
#define LIBTENSOR_BTOD_SET_ELEM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_set_elem class

    \ingroup libtensor_tests_btod
**/
class btod_set_elem_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_ELEM_TEST_H
