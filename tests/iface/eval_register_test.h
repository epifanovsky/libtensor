#ifndef LIBTENSOR_EVAL_REGISTER_TEST_H
#define LIBTENSOR_EVAL_REGISTER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::eval_register class

    \ingroup libtensor_tests_iface
**/
class eval_register_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();

};


} // namespace libtensor


#endif // LIBTENSOR_EVAL_REGISTER_TEST_H
