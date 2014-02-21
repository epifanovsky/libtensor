#ifndef LIBTENSOR_ANY_TENSOR_TEST_H
#define LIBTENSOR_ANY_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::any_tensor class

    \ingroup libtensor_tests_iface
**/
class any_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_ANY_TENSOR_TEST_H

