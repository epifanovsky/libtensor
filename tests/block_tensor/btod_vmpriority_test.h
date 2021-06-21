#ifndef LIBTENSOR_BTOD_VMPRIORITY_TEST_H
#define LIBTENSOR_BTOD_VMPRIORITY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_vmpriority class

    \ingroup libtensor_tests_btod
 **/
class btod_vmpriority_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_VMPRIORITY_TEST_H
